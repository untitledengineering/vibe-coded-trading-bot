"""Aggregate 1-minute OHLCV bars into 5-minute bars.

NSE pre-open starts at 09:00, regular session 09:15–15:30 IST. 09:15 IST
falls exactly on a 5-minute boundary in both IST and UTC (09:15 IST = 03:45
UTC, both multiples of 5 mins from epoch), so floor-by-300-seconds gives the
correct buckets without any timezone-aware bucketing.
"""

from __future__ import annotations

import pandas as pd

FIVE_MIN_SECONDS = 300


def aggregate_to_5m(bars_1m: pd.DataFrame) -> pd.DataFrame:
    """Roll 1-minute OHLCV into 5-minute OHLCV. Empty input -> empty output.

    Input columns required: minute_ts, open, high, low, close, volume.
    Output uses the same column names; minute_ts is the start of the 5-min bucket.
    """
    if bars_1m.empty:
        return bars_1m.iloc[0:0].copy()
    df = bars_1m[["minute_ts", "open", "high", "low", "close", "volume"]].copy()
    df["bucket_ts"] = (df["minute_ts"] // FIVE_MIN_SECONDS) * FIVE_MIN_SECONDS
    agg = (
        df.groupby("bucket_ts", as_index=False)
        .agg(open=("open", "first"),
             high=("high", "max"),
             low=("low", "min"),
             close=("close", "last"),
             volume=("volume", "sum"))
        .rename(columns={"bucket_ts": "minute_ts"})
        .sort_values("minute_ts")
        .reset_index(drop=True)
    )
    return agg
