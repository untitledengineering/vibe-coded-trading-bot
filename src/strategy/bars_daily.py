"""Aggregate 1-minute bars into IST daily bars.

A daily bar covers all 1-min bars sharing the same IST trading session date.
We use (minute_ts + IST_OFFSET) // 86400 — same convention as the rest of the
codebase — so 09:15..15:30 IST falls into one bucket.
"""

from __future__ import annotations

import pandas as pd

from src.features.technical import IST_OFFSET_SECONDS, SECONDS_PER_DAY


def aggregate_to_daily(bars_1m: pd.DataFrame) -> pd.DataFrame:
    """Roll 1-minute OHLCV into daily OHLCV. Empty input -> empty output.

    Input must have: minute_ts, open, high, low, close, volume
    Output: ist_day (int days since epoch in IST), open/high/low/close/volume
    """
    if bars_1m.empty:
        return bars_1m.iloc[0:0].copy()
    df = bars_1m[["minute_ts", "open", "high", "low", "close", "volume"]].copy()
    df["ist_day"] = (df["minute_ts"] + IST_OFFSET_SECONDS) // SECONDS_PER_DAY
    return (
        df.groupby("ist_day", as_index=False)
        .agg(open=("open", "first"),
             high=("high", "max"),
             low=("low", "min"),
             close=("close", "last"),
             volume=("volume", "sum"))
        .sort_values("ist_day")
        .reset_index(drop=True)
    )


def ist_day_now(now_ts: float) -> int:
    """Today's ist_day integer at the given epoch second."""
    return int((now_ts + IST_OFFSET_SECONDS) // SECONDS_PER_DAY)
