"""CLI: compute and inspect technical features for one symbol from bars_1m.

    python -m src.features --symbol RELIANCE
    python -m src.features --symbol RELIANCE --tail 20
"""

import argparse
import sqlite3
import sys

import pandas as pd

from src.data.universe import load_universe
from src.features.technical import FEATURE_COLUMNS, compute_features
from src.utils.config import DB_PATH


def _resolve_instrument_key(symbol: str) -> str:
    universe = load_universe()
    matches = [u for u in universe if u["trading_symbol"].upper() == symbol.upper()]
    if not matches:
        raise SystemExit(f"Symbol {symbol!r} not in universe (data/universe.json).")
    return matches[0]["instrument_key"]


def _load_bars(instrument_key: str) -> pd.DataFrame:
    # Read-only connection. The backfill may hold a write lock; this still works
    # against the WAL but errors cleanly if SQLite is busy.
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=10)
    try:
        df = pd.read_sql_query(
            """
            SELECT minute_ts, open, high, low, close, volume
            FROM bars_1m
            WHERE instrument_key = ?
            ORDER BY minute_ts ASC
            """,
            conn,
            params=(instrument_key,),
        )
    finally:
        conn.close()
    return df


def main():
    parser = argparse.ArgumentParser(description="Inspect computed features for one symbol.")
    parser.add_argument("--symbol", required=True, help="Trading symbol, e.g. RELIANCE")
    parser.add_argument("--tail", type=int, default=5, help="Print the last N rows")
    args = parser.parse_args()

    key = _resolve_instrument_key(args.symbol)
    bars = _load_bars(key)
    if bars.empty:
        print(f"No bars in bars_1m for {args.symbol} ({key}). Run the backfill first.")
        sys.exit(1)

    feats = compute_features(bars)

    print(f"Symbol:        {args.symbol}")
    print(f"Instrument:    {key}")
    print(f"Bars loaded:   {len(feats):,}")
    print(f"Range:         {pd.to_datetime(feats['minute_ts'].min(), unit='s')} .. "
          f"{pd.to_datetime(feats['minute_ts'].max(), unit='s')}")
    print()
    print("NaN counts per feature (lower is better — high count means short history):")
    for col in FEATURE_COLUMNS:
        print(f"  {col:10s}  {feats[col].isna().sum():>8,} of {len(feats):,}")
    print()
    print(f"Tail ({args.tail} rows):")
    cols = ["minute_ts", "close", "volume", *FEATURE_COLUMNS]
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(feats[cols].tail(args.tail).to_string(index=False))


if __name__ == "__main__":
    main()
