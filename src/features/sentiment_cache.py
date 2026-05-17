"""Per-symbol live sentiment score derived from the news table.

Design
------
We don't retrain the model with sentiment — we don't have historical scored
headlines. Instead, sentiment is used as a POST-MODEL VETO in decide():

    strong bearish news  →  block new long entries for that symbol
    strong bullish news  →  block new short entries for that symbol

Score formula per symbol
------------------------
For every scored headline relevant to the symbol (or market-wide):

    weight_i = confidence_i * exp(-age_minutes_i / decay_minutes_i)
    weighted_score = Σ(score_i * weight_i) / Σ(weight_i)

A headline whose decay window has elapsed contributes weight ≈ 0 and is
effectively ignored. Market-wide headlines count at MARKET_WEIGHT_FACTOR of
a symbol-specific headline.

Returns {} when the news table is empty or all headlines are unscored.
"""

from __future__ import annotations

import math
import sqlite3
import time
from typing import Dict, List, Optional

from src.utils.config import DB_PATH

LOOKBACK_SECONDS = 7_200      # consider headlines from the past 2 hours
MARKET_WEIGHT_FACTOR = 0.4    # market-wide headlines count less than ticker-specific ones


def _fetch_recent_scored(
    conn: sqlite3.Connection,
    instrument_keys: List[str],
    now_ts: int,
    lookback_seconds: int,
) -> list:
    """Return rows (instrument_key, score, confidence, decay_minutes, published_at)
    for headlines published within the lookback window that have been scored.
    Includes market-wide rows (instrument_key IS NULL)."""
    since = now_ts - lookback_seconds
    placeholders = ",".join("?" * len(instrument_keys))
    cur = conn.execute(
        f"""
        SELECT instrument_key, sentiment_score, sentiment_confidence,
               sentiment_decay_minutes, published_at
        FROM news
        WHERE sentiment_at IS NOT NULL
          AND published_at >= ?
          AND (instrument_key IN ({placeholders}) OR instrument_key IS NULL)
        """,
        (since, *instrument_keys),
    )
    return cur.fetchall()


def live_sentiment_by_symbol(
    instrument_keys: List[str],
    now_ts: Optional[int] = None,
    lookback_seconds: int = LOOKBACK_SECONDS,
    db_path: Optional[str] = None,
) -> Dict[str, float]:
    """Compute a weighted sentiment score per symbol.

    Returns a dict mapping instrument_key → float in [-1, 1].
    Symbols with no relevant scored headlines are absent from the dict —
    the caller treats absence as neutral (0.0).
    """
    if not instrument_keys:
        return {}
    ts = now_ts if now_ts is not None else int(time.time())

    try:
        conn = sqlite3.connect(
            f"file:{db_path or DB_PATH}?mode=ro", uri=True, timeout=5
        )
    except sqlite3.OperationalError:
        return {}

    try:
        rows = _fetch_recent_scored(conn, instrument_keys, ts, lookback_seconds)
    finally:
        conn.close()

    if not rows:
        return {}

    # Accumulate weighted score per symbol.
    # Market-wide rows (instrument_key IS NULL) contribute to every symbol at
    # MARKET_WEIGHT_FACTOR of their natural weight.
    weight_sum: Dict[str, float] = {k: 0.0 for k in instrument_keys}
    score_sum: Dict[str, float] = {k: 0.0 for k in instrument_keys}

    for (ikey, score, confidence, decay_min, published_at) in rows:
        if score is None or confidence is None or decay_min is None:
            continue
        age_minutes = max(0.0, (ts - published_at) / 60.0)
        decay = max(1, decay_min)  # guard against 0
        weight = float(confidence) * math.exp(-age_minutes / decay)
        if weight < 1e-6:
            continue  # negligible — skip

        if ikey is None:
            # Market-wide: apply to every symbol at reduced weight.
            for k in instrument_keys:
                w = weight * MARKET_WEIGHT_FACTOR
                score_sum[k] += float(score) * w
                weight_sum[k] += w
        else:
            score_sum[ikey] += float(score) * weight
            weight_sum[ikey] += weight

    result: Dict[str, float] = {}
    for k in instrument_keys:
        if weight_sum[k] > 1e-9:
            raw = score_sum[k] / weight_sum[k]
            result[k] = max(-1.0, min(1.0, raw))

    return result
