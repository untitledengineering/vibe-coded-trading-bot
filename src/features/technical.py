"""Technical feature engineering on 1-minute OHLCV bars.

Designed so the SAME function works during model training (batch over historical
bars_1m) and live inference (rolling buffer of bars_live). Train/serve skew is
the easiest way to ship a quietly broken trading model — keep that in mind
before refactoring.

Input contract:
    DataFrame for a SINGLE instrument, sorted ascending by minute_ts, with
    columns: minute_ts (int, epoch seconds), open, high, low, close, volume.

Output:
    The input DataFrame with these feature columns appended:
        ret_5m, ret_15m, ret_30m
        rsi_14
        atr_14
        vwap_dev
        vol_z_20d        (volume z-score vs prior 20 days of the same minute-of-day)
        gap_pct          (today's open vs prior session's close; constant within session)

Bars with insufficient history get NaN in the relevant feature; the caller
decides whether to dropna before training / inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# IST is UTC+5:30. NSE trading window: 09:15..15:30 IST.
IST_OFFSET_SECONDS = 5 * 3600 + 30 * 60
SECONDS_PER_DAY = 86400

REQUIRED_COLUMNS = ("minute_ts", "open", "high", "low", "close", "volume")


# ---------- Primitives ----------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. Returns 0..100, NaN until ``period`` warmup bars accumulate."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing is EWM with alpha = 1/period and adjust=False.
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When there's no loss at all, RSI is 100 by convention.
    rsi = rsi.where(avg_loss != 0, other=100.0)
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's ATR. NaN until warmup."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _session_date(minute_ts: pd.Series) -> pd.Series:
    """Integer 'days since epoch' in IST. Equal within one NSE trading session."""
    return ((minute_ts + IST_OFFSET_SECONDS) // SECONDS_PER_DAY).astype("int64")


def _session_vwap(typical_price: pd.Series, volume: pd.Series, session_date: pd.Series) -> pd.Series:
    """Cumulative VWAP, reset at each session boundary. NaN if no volume yet."""
    tp_vol = typical_price * volume
    csum_tpv = tp_vol.groupby(session_date).cumsum()
    csum_vol = volume.groupby(session_date).cumsum()
    return csum_tpv / csum_vol.where(csum_vol > 0, other=np.nan)


def _minute_of_day(minute_ts: pd.Series) -> pd.Series:
    """0..1439 — which minute of the IST day this bar belongs to. Stable bucket key
    for volume seasonality (09:15 IST always rolls into the same bucket)."""
    ist_seconds = (minute_ts + IST_OFFSET_SECONDS) % SECONDS_PER_DAY
    return (ist_seconds // 60).astype("int64")


def _vol_zscore_seasonal(volume: pd.Series, minute_of_day: pd.Series, window: int = 20) -> pd.Series:
    """For each row, compute z-score of `volume` against the prior `window` occurrences
    of the same minute-of-day. We .shift(1) so the current bar is never in its own stats.

    Vectorized via pivot: avoids 375 Python groupby.apply() calls per symbol.
    The mathematics are identical to the original per-group rolling approach.
    """
    occ = minute_of_day.groupby(minute_of_day).cumcount() if False else (
        pd.Series(minute_of_day.values).groupby(minute_of_day.values).cumcount().values
    )
    # pivot: rows = occurrence index, cols = minute_of_day value
    wide = (
        pd.DataFrame({"vol": volume.values, "mod": minute_of_day.values, "occ": occ})
        .pivot_table(index="occ", columns="mod", values="vol", aggfunc="first")
    )
    shifted = wide.shift(1)
    roll_mean = shifted.rolling(window, min_periods=window).mean()
    roll_std  = shifted.rolling(window, min_periods=window).std(ddof=0)
    z_wide = (wide - roll_mean) / roll_std.where(roll_std > 0, other=np.nan)

    # Vectorised map-back: find (occ, col) for every original row
    z_arr = z_wide.to_numpy()
    mod_labels = z_wide.columns.to_numpy()
    col_idx = np.searchsorted(mod_labels, minute_of_day.values)
    col_idx = np.clip(col_idx, 0, len(mod_labels) - 1)
    occ_idx = np.clip(occ, 0, z_arr.shape[0] - 1)
    result = z_arr[occ_idx, col_idx]
    # Mask any col mismatch (minute_of_day value not in pivot — shouldn't happen)
    mismatch = mod_labels[col_idx] != minute_of_day.values
    result[mismatch] = np.nan
    return pd.Series(result, index=volume.index)


def _gap_pct(open_: pd.Series, close: pd.Series, session_date: pd.Series) -> pd.Series:
    """Per-session gap: (today_open - prev_session_close) / prev_session_close.
    Same value for every minute within a session."""
    first_open = open_.groupby(session_date).transform("first")
    last_close_by_session = close.groupby(session_date).last()
    prev_close_by_session = last_close_by_session.shift(1)
    prev_close = session_date.map(prev_close_by_session)
    return (first_open - prev_close) / prev_close.where(prev_close > 0, other=np.nan)


# ---------- Public entry point ----------

def compute_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Append feature columns. See module docstring for the input contract."""
    missing = [c for c in REQUIRED_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"compute_features missing required columns: {missing}")
    if len(bars) == 0:
        return bars.assign(
            ret_5m=pd.Series(dtype="float64"),
            ret_15m=pd.Series(dtype="float64"),
            ret_30m=pd.Series(dtype="float64"),
            rsi_14=pd.Series(dtype="float64"),
            atr_14=pd.Series(dtype="float64"),
            vwap_dev=pd.Series(dtype="float64"),
            vol_z_20d=pd.Series(dtype="float64"),
            gap_pct=pd.Series(dtype="float64"),
        )

    # Defensive: ensure ascending order on minute_ts. Callers should already do this,
    # but a single mis-sorted batch would silently corrupt every diff/rolling op.
    if not bars["minute_ts"].is_monotonic_increasing:
        bars = bars.sort_values("minute_ts").reset_index(drop=True)
    else:
        bars = bars.reset_index(drop=True)

    out = bars.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    open_ = out["open"]
    volume = out["volume"]

    session_date = _session_date(out["minute_ts"])
    minute_of_day = _minute_of_day(out["minute_ts"])

    out["ret_5m"] = close.pct_change(5)
    out["ret_15m"] = close.pct_change(15)
    out["ret_30m"] = close.pct_change(30)
    out["rsi_14"] = _rsi(close, period=14)
    out["atr_14"] = _atr(high, low, close, period=14)

    typical_price = (high + low + close) / 3.0
    vwap = _session_vwap(typical_price, volume, session_date)
    out["vwap_dev"] = (close - vwap) / vwap

    out["vol_z_20d"] = _vol_zscore_seasonal(volume, minute_of_day, window=20)
    out["gap_pct"] = _gap_pct(open_, close, session_date)

    return out


FEATURE_COLUMNS = (
    "ret_5m",
    "ret_15m",
    "ret_30m",
    "rsi_14",
    "atr_14",
    "vwap_dev",
    "vol_z_20d",
    "gap_pct",
)
