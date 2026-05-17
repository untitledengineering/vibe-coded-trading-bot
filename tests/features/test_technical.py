"""Synthetic-data tests for the technical feature pipeline.

These tests intentionally avoid hitting bars_1m so the suite doesn't need a
populated DB. Real-data validation is a manual `python -m src.features` step
after the backfill completes.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.features import technical
from src.features.technical import FEATURE_COLUMNS, compute_features

IST_OFFSET = technical.IST_OFFSET_SECONDS
DAY = 86400


def _session_minutes(date_epoch_day: int, count: int, start_minute_ist: int = 555):
    """Generate `count` consecutive minute_ts values starting at 09:15 IST on the
    given IST day. start_minute_ist defaults to 9*60 + 15 = 555 (NSE open)."""
    day_start_utc = date_epoch_day * DAY - IST_OFFSET
    base = day_start_utc + start_minute_ist * 60
    return [base + 60 * i for i in range(count)]


def _make_session_df(date_epoch_day: int, closes: list[float], volumes: list[int] = None):
    """Build a DF for one session with the given close sequence.
    open == close (no intrabar movement), high/low = close ± 0.5 for ATR sanity."""
    n = len(closes)
    minute_ts = _session_minutes(date_epoch_day, n)
    closes = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "minute_ts": minute_ts,
            "open": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": volumes if volumes else [100] * n,
        }
    )


# ----- contract -----

def test_missing_columns_raises():
    bad = pd.DataFrame({"minute_ts": [1, 2], "close": [1.0, 2.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        compute_features(bad)


def test_empty_input_returns_empty_with_feature_columns():
    empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in technical.REQUIRED_COLUMNS})
    out = compute_features(empty)
    assert len(out) == 0
    for col in FEATURE_COLUMNS:
        assert col in out.columns


def test_sorts_unsorted_input():
    df = _make_session_df(20000, [100, 101, 102, 103, 104])
    shuffled = df.iloc[[3, 0, 4, 1, 2]].reset_index(drop=True)
    out = compute_features(shuffled)
    assert out["minute_ts"].is_monotonic_increasing


# ----- returns -----

def test_ret_5m_is_exact_for_known_close_sequence():
    closes = [100.0] + [100.0 + i for i in range(1, 11)]  # 100,101,...,110 (11 bars)
    df = _make_session_df(20000, closes)
    out = compute_features(df)
    # ret_5m at index 5: (close[5] - close[0]) / close[0] = (105 - 100) / 100 = 0.05
    assert out["ret_5m"].iloc[5] == pytest.approx(0.05)
    # NaN until we have 5 bars of history.
    assert out["ret_5m"].iloc[:5].isna().all()


# ----- RSI -----

def test_rsi_of_monotonic_increase_is_100():
    closes = list(range(100, 150))
    df = _make_session_df(20000, closes)
    out = compute_features(df)
    # After warmup (14 bars), RSI should saturate at ~100 because there are no losses.
    final_rsi = out["rsi_14"].iloc[-1]
    assert final_rsi == pytest.approx(100.0, rel=1e-3)


def test_rsi_of_monotonic_decrease_is_zero():
    closes = list(range(150, 100, -1))
    df = _make_session_df(20000, closes)
    out = compute_features(df)
    final_rsi = out["rsi_14"].iloc[-1]
    assert final_rsi == pytest.approx(0.0, abs=1e-6)


def test_rsi_is_nan_during_warmup():
    closes = list(range(100, 110))  # only 10 bars, less than period=14
    df = _make_session_df(20000, closes)
    out = compute_features(df)
    assert out["rsi_14"].isna().all()


# ----- ATR -----

def test_atr_zero_for_flat_bars():
    closes = [100.0] * 30
    df = pd.DataFrame(
        {
            "minute_ts": _session_minutes(20000, 30),
            "open": closes, "high": closes, "low": closes, "close": closes,
            "volume": [0] * 30,
        }
    )
    out = compute_features(df)
    assert out["atr_14"].dropna().abs().max() == 0


def test_atr_positive_when_range_present():
    df = _make_session_df(20000, list(range(100, 130)))
    out = compute_features(df)
    assert (out["atr_14"].dropna() > 0).all()


# ----- VWAP -----

def test_vwap_dev_zero_when_price_is_constant():
    df = _make_session_df(20000, [100.0] * 20, volumes=[100] * 20)
    out = compute_features(df)
    # With flat prices, VWAP equals the price, so deviation is 0.
    assert out["vwap_dev"].dropna().abs().max() == pytest.approx(0.0, abs=1e-9)


def test_vwap_dev_signs_match_price_movement_relative_to_vwap():
    # Prices climbing: late bars are above the running VWAP -> positive deviation.
    closes = list(range(100, 120))
    df = _make_session_df(20000, closes, volumes=[100] * 20)
    out = compute_features(df)
    # First valid VWAP dev should be 0 (only one bar in cumsum), last should be positive.
    assert out["vwap_dev"].iloc[0] == pytest.approx(0.0)
    assert out["vwap_dev"].iloc[-1] > 0


# ----- Volume z-score -----

def test_vol_z_20d_returns_nan_until_20_prior_observations_at_same_minute():
    # Build 25 sessions, each 1 bar long, all at the same minute-of-day.
    closes = [100.0]
    sessions = []
    for day in range(20000, 20025):
        sessions.append(_make_session_df(day, closes, volumes=[100]))
    df = pd.concat(sessions, ignore_index=True)
    out = compute_features(df)
    # First 20 bars: NaN (not enough prior observations of this minute-of-day).
    assert out["vol_z_20d"].iloc[:20].isna().all()
    # 21st bar onwards: defined (z-score of constant volume is undefined -> NaN
    # because std == 0). The function returns NaN, not inf.
    assert out["vol_z_20d"].iloc[20:].isna().all()


def test_vol_z_20d_spikes_on_anomaly():
    # 20 sessions of mildly-varying volume (so std > 0), then a 21st with a huge spike.
    rng = np.random.default_rng(seed=42)
    history_vols = (100 + rng.normal(0, 5, size=20)).round().astype(int).tolist()
    rows = [_make_session_df(20000 + i, [100.0], volumes=[history_vols[i]]) for i in range(20)]
    rows.append(_make_session_df(20020, [100.0], volumes=[1000]))  # spike
    df = pd.concat(rows, ignore_index=True)
    out = compute_features(df)
    spike_z = out["vol_z_20d"].iloc[-1]
    assert not math.isnan(spike_z), f"expected definite z-score, got NaN"
    # Spike is ~180 std deviations above the historical mean given our synthetic noise.
    assert spike_z > 3


# ----- Gap -----

def test_gap_pct_within_session_is_constant():
    df1 = _make_session_df(20000, [100, 100, 100], volumes=[100] * 3)
    df2 = _make_session_df(20001, [105, 105, 105], volumes=[100] * 3)
    df = pd.concat([df1, df2], ignore_index=True)
    out = compute_features(df)
    # Day 1: no prior session -> NaN.
    assert out["gap_pct"].iloc[:3].isna().all()
    # Day 2: open=105, prev close=100 -> 5% gap. Same value across the whole session.
    day2 = out["gap_pct"].iloc[3:]
    assert np.allclose(day2.values, 0.05)


def test_gap_pct_handles_three_sessions():
    df1 = _make_session_df(20000, [100], volumes=[100])
    df2 = _make_session_df(20001, [110], volumes=[100])  # +10%
    df3 = _make_session_df(20002, [99], volumes=[100])   # -10%
    df = pd.concat([df1, df2, df3], ignore_index=True)
    out = compute_features(df)
    assert math.isnan(out["gap_pct"].iloc[0])
    assert out["gap_pct"].iloc[1] == pytest.approx(0.10)
    assert out["gap_pct"].iloc[2] == pytest.approx(-0.10)


# ----- Output schema -----

def test_all_feature_columns_present():
    df = _make_session_df(20000, list(range(100, 120)))
    out = compute_features(df)
    for col in FEATURE_COLUMNS:
        assert col in out.columns, f"missing feature column: {col}"
    # Original columns preserved.
    for col in technical.REQUIRED_COLUMNS:
        assert col in out.columns
