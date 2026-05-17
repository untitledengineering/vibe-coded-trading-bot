"""Live feature builder for the paper engine.

Single-pass batch read: one SQL query pulls the last 40 calendar days of bars
for ALL symbols at once, then pandas groupby computes features per symbol.
This scales to 209 symbols without meaningful latency (~0.5s vs 30s+ for the
naive per-symbol approach).

40 calendar days ≈ 28 trading days × 375 min/day ≈ 10,500 bars per symbol,
which is enough warmup for the most demanding feature (vol_z_20d needs 20d).
"""

from __future__ import annotations

import sqlite3
import time
from typing import List, Optional

import pandas as pd

from src.features.technical import FEATURE_COLUMNS, compute_features
from src.utils.config import DB_PATH

LOOKBACK_CALENDAR_DAYS = 40
LOOKBACK_SECONDS = LOOKBACK_CALENDAR_DAYS * 86_400


def _read_all_bars(
    conn: sqlite3.Connection,
    instrument_keys: List[str],
    cutoff_ts: int,
) -> pd.DataFrame:
    """One SQL read: historical bars since cutoff + today's live bars, all symbols."""
    placeholders = ",".join("?" * len(instrument_keys))

    historical = pd.read_sql_query(
        f"""
        SELECT instrument_key, minute_ts, open, high, low, close, volume
        FROM bars_1m
        WHERE instrument_key IN ({placeholders})
          AND minute_ts >= ?
        ORDER BY instrument_key, minute_ts ASC
        """,
        conn,
        params=(*instrument_keys, cutoff_ts),
    )

    live = pd.read_sql_query(
        f"""
        SELECT instrument_key, minute_ts, open, high, low, close, volume
        FROM bars_live
        WHERE instrument_key IN ({placeholders})
        ORDER BY instrument_key, minute_ts ASC
        """,
        conn,
        params=instrument_keys,
    )

    if historical.empty and live.empty:
        return pd.DataFrame()
    if live.empty:
        return historical
    if historical.empty:
        return live

    # live bars win on timestamp conflicts (same symbol + minute_ts).
    live_idx = set(zip(live["instrument_key"], live["minute_ts"]))
    mask = [
        (ik, mt) not in live_idx
        for ik, mt in zip(historical["instrument_key"], historical["minute_ts"])
    ]
    historical = historical[mask]
    return (
        pd.concat([historical, live], ignore_index=True)
        .sort_values(["instrument_key", "minute_ts"])
        .reset_index(drop=True)
    )


def build_live_feature_frame(
    instrument_keys: List[str],
    db_path: Optional[str] = None,
) -> pd.DataFrame:
    """One-row-per-symbol feature frame for the decision engine.

    Reads all symbols in a single SQL round-trip and computes features via
    groupby. Symbols that lack enough history for warmup are silently dropped.

    Output columns: instrument_key, close, <FEATURE_COLUMNS...>
    """
    if not instrument_keys:
        return pd.DataFrame()

    cutoff_ts = int(time.time()) - LOOKBACK_SECONDS
    try:
        conn = sqlite3.connect(
            f"file:{db_path or DB_PATH}?mode=ro", uri=True, timeout=10
        )
    except sqlite3.OperationalError:
        return pd.DataFrame()

    try:
        all_bars = _read_all_bars(conn, instrument_keys, cutoff_ts)
    finally:
        conn.close()

    if all_bars.empty:
        return pd.DataFrame()

    rows = []
    for key, group in all_bars.groupby("instrument_key", sort=False):
        group = group.sort_values("minute_ts").reset_index(drop=True)
        feats = compute_features(group)
        usable = feats.dropna(subset=list(FEATURE_COLUMNS))
        if usable.empty:
            continue
        latest = usable.iloc[-1]
        row = {"instrument_key": key, "close": float(latest["close"])}
        for col in FEATURE_COLUMNS:
            row[col] = float(latest[col])
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
