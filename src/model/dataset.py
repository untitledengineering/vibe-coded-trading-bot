"""Assemble the training matrix (X, y) from bars_1m across the F&O universe.

Iterates per-symbol (cheap query, modest memory), computes features and labels,
drops warmup/boundary NaNs, and yields per-symbol frames. The trainer then
concatenates them. We never load all 17M raw bars into memory at once.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd

from src.data.universe import load_universe
from src.features.technical import FEATURE_COLUMNS, compute_features
from src.model.labels import DEFAULT_HORIZON_MINUTES, attach_labels
from src.utils.config import DB_PATH
from src.utils.logger import logger


@dataclass(frozen=True)
class DatasetSpec:
    feature_columns: tuple = FEATURE_COLUMNS
    label_horizon_minutes: int = DEFAULT_HORIZON_MINUTES

    @property
    def label_regression_column(self) -> str:
        return f"fwd_ret_{self.label_horizon_minutes}m"

    @property
    def label_classification_column(self) -> str:
        return f"fwd_up_{self.label_horizon_minutes}m"


def _read_bars_for_symbol(conn: sqlite3.Connection, instrument_key: str) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT minute_ts, open, high, low, close, volume
        FROM bars_1m
        WHERE instrument_key = ?
        ORDER BY minute_ts ASC
        """,
        conn,
        params=(instrument_key,),
    )


def build_dataset_for_symbol(
    bars: pd.DataFrame,
    instrument_key: str,
    spec: DatasetSpec = DatasetSpec(),
) -> pd.DataFrame:
    """One symbol → one ready-to-train frame. Drops warmup/boundary NaNs.

    Columns of the returned frame:
        instrument_key, minute_ts,
        <feature_columns ...>,
        <regression label>, <classification label>
    """
    if bars.empty:
        return pd.DataFrame()

    feats = compute_features(bars)
    labelled = attach_labels(feats, horizon_minutes=spec.label_horizon_minutes)

    # Raw bar columns are preserved so the backtester can check SL/TP triggers
    # against intraday high/low. Training is unaffected — x_y_arrays() filters
    # back down to the feature columns from the spec.
    keep_cols = (
        ["minute_ts", "open", "high", "low", "close", "volume"]
        + list(spec.feature_columns)
        + [spec.label_regression_column, spec.label_classification_column]
    )
    out = labelled[keep_cols].copy()
    out.insert(0, "instrument_key", instrument_key)

    # Drop rows where any required column is NaN. This kills the warmup window
    # AND the trailing HORIZON minutes of each session in one shot.
    required = list(spec.feature_columns) + [spec.label_regression_column]
    out = out.dropna(subset=required).reset_index(drop=True)
    return out


def iter_dataset(
    db_path: Optional[str] = None,
    spec: DatasetSpec = DatasetSpec(),
    symbols: Optional[List[str]] = None,
) -> Iterator[pd.DataFrame]:
    """Yield one ready-to-train DataFrame per symbol. Caller concatenates."""
    db_path = db_path or DB_PATH
    universe = load_universe()
    if symbols:
        wanted = {s.upper() for s in symbols}
        universe = [u for u in universe if u["trading_symbol"].upper() in wanted]

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    try:
        for item in universe:
            key = item["instrument_key"]
            bars = _read_bars_for_symbol(conn, key)
            if bars.empty:
                logger.warning(f"No bars for {item['trading_symbol']} ({key}); skipping")
                continue
            frame = build_dataset_for_symbol(bars, key, spec=spec)
            if not frame.empty:
                yield frame
    finally:
        conn.close()


def assemble_full_dataset(
    db_path: Optional[str] = None,
    spec: DatasetSpec = DatasetSpec(),
    symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Materialise the entire (X, y) frame in memory. ~1 GB for the full universe.
    Use iter_dataset() instead if you want streaming."""
    parts = list(iter_dataset(db_path=db_path, spec=spec, symbols=symbols))
    if not parts:
        return pd.DataFrame()
    full = pd.concat(parts, ignore_index=True)
    logger.info(
        f"Dataset assembled: {len(full):,} rows across "
        f"{full['instrument_key'].nunique()} symbols"
    )
    return full


def time_split(
    df: pd.DataFrame,
    train_end_ts: int,
    test_end_ts: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict temporal split. Train = minute_ts <= train_end_ts.
    Test = train_end_ts < minute_ts <= test_end_ts (or to end if test_end_ts None)."""
    train = df[df["minute_ts"] <= train_end_ts]
    if test_end_ts is None:
        test = df[df["minute_ts"] > train_end_ts]
    else:
        test = df[(df["minute_ts"] > train_end_ts) & (df["minute_ts"] <= test_end_ts)]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def x_y_arrays(
    df: pd.DataFrame,
    spec: DatasetSpec = DatasetSpec(),
    label: str = "regression",
) -> tuple[np.ndarray, np.ndarray]:
    """Pull X (features) and y (label) as numpy arrays for the trainer."""
    X = df[list(spec.feature_columns)].to_numpy(dtype=np.float32)
    if label == "regression":
        y = df[spec.label_regression_column].to_numpy(dtype=np.float32)
    elif label == "classification":
        y = df[spec.label_classification_column].to_numpy(dtype=np.float32)
    else:
        raise ValueError(f"label must be 'regression' or 'classification', got {label!r}")
    return X, y
