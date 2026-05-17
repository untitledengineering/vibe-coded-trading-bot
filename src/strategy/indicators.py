"""Technical indicators for the rule-based intraday strategy.

VWAP — reused from features.technical so the math is identical to what the
ML pipeline computes.

Supertrend(period=7, multiplier=3) — implemented here. Recursive band logic
means we use an explicit loop; pandas-native vectorisation would only
obscure the formula and the input sizes (≤2000 5-min bars per symbol per
day) make speed a non-issue.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.technical import _atr, _session_vwap, _session_date


def session_vwap(bars: pd.DataFrame) -> pd.Series:
    """Cumulative session VWAP from minute_ts/high/low/close/volume bars."""
    typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    session = _session_date(bars["minute_ts"])
    return _session_vwap(typical_price, bars["volume"], session)


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 7,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """Classic Supertrend. Returns (supertrend_line, direction).

    direction encodes the trend:
        +1 = uptrend (bullish), supertrend_line is the lower band (support)
        -1 = downtrend (bearish), supertrend_line is the upper band (resistance)

    NaN until ATR warmup completes (`period` bars). Caller filters NaNs.
    """
    if not (len(high) == len(low) == len(close)):
        raise ValueError("Supertrend: high/low/close must be the same length")
    n = len(close)
    if n == 0:
        empty = pd.Series([], dtype="float64")
        return empty, empty

    atr = _atr(high, low, close, period=period)
    hl_avg = (high + low) / 2.0
    upper_basic = hl_avg + multiplier * atr
    lower_basic = hl_avg - multiplier * atr

    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    direction = np.zeros(n, dtype=np.int8)
    st = np.full(n, np.nan)

    # Find the first bar where ATR is defined; that's our starting point.
    first_valid = atr.first_valid_index()
    if first_valid is None:
        return pd.Series(st, index=close.index, name="supertrend"), \
               pd.Series(direction, index=close.index, name="supertrend_dir")
    start = close.index.get_loc(first_valid)

    upper[start] = upper_basic.iloc[start]
    lower[start] = lower_basic.iloc[start]
    direction[start] = 1  # start as uptrend; first valid direction is symmetric
    st[start] = lower[start]

    for i in range(start + 1, n):
        ub = upper_basic.iloc[i]
        lb = lower_basic.iloc[i]
        prev_close = close.iloc[i - 1]
        # Trailing-band smoothing.
        if ub < upper[i - 1] or prev_close > upper[i - 1]:
            upper[i] = ub
        else:
            upper[i] = upper[i - 1]
        if lb > lower[i - 1] or prev_close < lower[i - 1]:
            lower[i] = lb
        else:
            lower[i] = lower[i - 1]
        # Direction flip detection.
        if close.iloc[i] > upper[i - 1]:
            direction[i] = 1
        elif close.iloc[i] < lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
        st[i] = lower[i] if direction[i] == 1 else upper[i]

    return (
        pd.Series(st, index=close.index, name="supertrend"),
        pd.Series(direction, index=close.index, name="supertrend_dir"),
    )
