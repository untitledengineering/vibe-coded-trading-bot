"""Synthetic-bar tests for label generation. The session-boundary handling is
the load-bearing piece — if it ever silently includes overnight gaps we end up
training the model on the wrong target."""

import numpy as np
import pandas as pd

from src.features.technical import IST_OFFSET_SECONDS
from src.model.labels import attach_labels, forward_return


def _session_minutes(day_index: int, count: int, start_minute_ist: int = 555):
    day_start_utc = day_index * 86400 - IST_OFFSET_SECONDS
    base = day_start_utc + start_minute_ist * 60
    return [base + 60 * i for i in range(count)]


def _make_bars(day_index: int, closes: list[float]):
    n = len(closes)
    closes_arr = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "minute_ts": _session_minutes(day_index, n),
            "open": closes_arr,
            "high": closes_arr + 0.1,
            "low": closes_arr - 0.1,
            "close": closes_arr,
            "volume": [100] * n,
        }
    )


def test_forward_return_exact_within_session():
    closes = [100.0 + i for i in range(30)]  # 100, 101, ..., 129
    bars = _make_bars(20000, closes)
    fwd = forward_return(bars, horizon_minutes=15)
    # At row 0: close[15] - close[0] / close[0] = (115-100)/100 = 0.15
    assert fwd.iloc[0] == 0.15
    # Last 15 rows have no t+15 within the session -> NaN
    assert fwd.iloc[-15:].isna().all()


def test_forward_return_nans_at_session_boundary():
    """Even with continuous minute_ts across two days, the label must NOT cross
    the trading session boundary."""
    closes_d1 = [100.0] * 30
    closes_d2 = [200.0] * 30
    bars = pd.concat(
        [_make_bars(20000, closes_d1), _make_bars(20001, closes_d2)],
        ignore_index=True,
    )
    fwd = forward_return(bars, horizon_minutes=15)
    # Last 15 bars of day 1 should be NaN (their t+15 falls in day 2).
    day1_tail = fwd.iloc[15:30]
    assert day1_tail.isna().all()
    # Day 2 should have valid labels for its first 15 bars.
    assert fwd.iloc[30:45].notna().all()


def test_forward_return_handles_missing_bars():
    """If a bar at t+15 simply doesn't exist (gap in history), label is NaN."""
    bars = _make_bars(20000, [100.0] * 10)
    # Drop the bar at row index 5 to create a gap.
    bars = bars.drop(index=[3]).reset_index(drop=True)
    fwd = forward_return(bars, horizon_minutes=15)
    # Every row's label requires bar at t+15, which doesn't exist anywhere in this
    # 9-bar frame, so all NaN.
    assert fwd.isna().all()


def test_forward_return_requires_sorted_input():
    bars = _make_bars(20000, [100.0, 101.0, 102.0])
    bars = bars.iloc[::-1].reset_index(drop=True)
    try:
        forward_return(bars)
    except ValueError as e:
        assert "sorted" in str(e).lower()
    else:
        raise AssertionError("expected ValueError on unsorted input")


def test_attach_labels_emits_both_columns():
    bars = _make_bars(20000, [100.0 + i for i in range(20)])
    out = attach_labels(bars, horizon_minutes=15)
    assert "fwd_ret_15m" in out.columns
    assert "fwd_up_15m" in out.columns
    # Monotonically increasing -> fwd return > 0 -> fwd_up == 1.
    valid = out.dropna(subset=["fwd_up_15m"])
    assert (valid["fwd_up_15m"] == 1.0).all()


def test_attach_labels_emits_zero_when_price_falls():
    bars = _make_bars(20000, [100.0 - i * 0.1 for i in range(20)])
    out = attach_labels(bars, horizon_minutes=15)
    valid = out.dropna(subset=["fwd_up_15m"])
    assert (valid["fwd_up_15m"] == 0.0).all()
