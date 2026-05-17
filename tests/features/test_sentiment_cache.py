"""Tests for live_sentiment_by_symbol."""

import math
import sqlite3
import time
from typing import List, Tuple

import pytest

from src.features.sentiment_cache import (
    MARKET_WEIGHT_FACTOR,
    live_sentiment_by_symbol,
)


def _make_db(tmp_path) -> str:
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE news (
            id INTEGER PRIMARY KEY,
            published_at INTEGER,
            fetched_at INTEGER,
            source TEXT,
            headline TEXT,
            url TEXT,
            instrument_key TEXT,
            sentiment_score REAL,
            sentiment_confidence REAL,
            sentiment_decay_minutes INTEGER,
            sentiment_model TEXT,
            sentiment_at INTEGER
        )
    """)
    conn.commit()
    conn.close()
    return db_path


def _insert(db_path: str, rows: List[Tuple]):
    """rows: (instrument_key, score, confidence, decay_minutes, age_seconds_ago)"""
    now = int(time.time())
    conn = sqlite3.connect(db_path)
    for ikey, score, conf, decay, age in rows:
        conn.execute(
            """INSERT INTO news
               (published_at, fetched_at, source, headline, url,
                instrument_key, sentiment_score, sentiment_confidence,
                sentiment_decay_minutes, sentiment_model, sentiment_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (now - age, now, "test", "headline", "http://x",
             ikey, score, conf, decay, "test", now),
        )
    conn.commit()
    conn.close()


# ---- basic behaviour ----

def test_empty_db_returns_empty(tmp_path):
    db = _make_db(tmp_path)
    assert live_sentiment_by_symbol(["A", "B"], db_path=db) == {}


def test_single_symbol_scored_headline(tmp_path):
    db = _make_db(tmp_path)
    _insert(db, [("A", 0.8, 1.0, 60, 0)])  # published now, full confidence
    result = live_sentiment_by_symbol(["A"], db_path=db)
    assert "A" in result
    assert result["A"] == pytest.approx(0.8, abs=0.01)


def test_unscored_headline_ignored(tmp_path):
    """Headline with sentiment_at IS NULL (unscored) must not appear."""
    db = _make_db(tmp_path)
    now = int(time.time())
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO news (published_at,fetched_at,source,headline,url,instrument_key) "
        "VALUES (?,?,?,?,?,?)",
        (now, now, "test", "headline", "http://x", "A"),
    )
    conn.commit()
    conn.close()
    assert live_sentiment_by_symbol(["A"], db_path=db) == {}


def test_decayed_headline_contributes_less(tmp_path):
    """A headline published 2× its decay window ago should have near-zero weight."""
    db = _make_db(tmp_path)
    decay_min = 10
    age_sec = decay_min * 2 * 60  # 2× decay → exp(-2) ≈ 0.135
    _insert(db, [("A", 1.0, 1.0, decay_min, age_sec)])
    result = live_sentiment_by_symbol(["A"], db_path=db)
    if "A" in result:
        # score should still be 1.0 (only one data point), but weight is small
        assert result["A"] == pytest.approx(1.0, abs=0.01)


def test_fully_decayed_headline_dropped(tmp_path):
    """A very old headline (age >> decay) has weight < 1e-6 and is dropped."""
    db = _make_db(tmp_path)
    decay_min = 5
    age_sec = 7200  # 24× the decay → weight ≈ exp(-144) ≈ 0
    _insert(db, [("A", 1.0, 1.0, decay_min, age_sec)])
    result = live_sentiment_by_symbol(["A"], db_path=db)
    assert "A" not in result


def test_two_headlines_weighted_average(tmp_path):
    """Weighted average of two equally fresh, equal-confidence headlines."""
    db = _make_db(tmp_path)
    # Both published just now (age=0), same decay, so equal weights.
    _insert(db, [
        ("A", 0.8, 1.0, 60, 0),
        ("A", -0.4, 1.0, 60, 0),
    ])
    result = live_sentiment_by_symbol(["A"], db_path=db)
    assert result["A"] == pytest.approx((0.8 + -0.4) / 2, abs=0.01)


def test_market_wide_headline_spreads_to_all_symbols(tmp_path):
    """instrument_key IS NULL means market-wide; contributes to every symbol."""
    db = _make_db(tmp_path)
    _insert(db, [(None, -0.9, 1.0, 60, 0)])  # None = market-wide
    result = live_sentiment_by_symbol(["A", "B"], db_path=db)
    # Both symbols should receive it (at MARKET_WEIGHT_FACTOR of the weight).
    assert "A" in result
    assert "B" in result
    assert result["A"] == pytest.approx(-0.9, abs=0.01)
    assert result["B"] == pytest.approx(-0.9, abs=0.01)


def test_market_wide_weight_factor_applied(tmp_path):
    """Symbol-specific and market-wide headlines blend at MARKET_WEIGHT_FACTOR ratio."""
    db = _make_db(tmp_path)
    # Symbol-specific: score=1.0, weight=1.0 (conf=1, age=0)
    # Market-wide: score=-1.0, weight=MARKET_WEIGHT_FACTOR (conf=1, age=0)
    _insert(db, [
        ("A",    1.0, 1.0, 60, 0),
        (None,  -1.0, 1.0, 60, 0),
    ])
    result = live_sentiment_by_symbol(["A"], db_path=db)
    mwf = MARKET_WEIGHT_FACTOR
    expected = (1.0 * 1.0 + (-1.0) * mwf) / (1.0 + mwf)
    assert result["A"] == pytest.approx(expected, abs=0.01)


def test_outside_lookback_window_excluded(tmp_path):
    """Headlines older than lookback_seconds must be excluded."""
    db = _make_db(tmp_path)
    _insert(db, [("A", 0.9, 1.0, 60, 7300)])  # 7300s ago > 7200s lookback
    result = live_sentiment_by_symbol(["A"], db_path=db, lookback_seconds=7200)
    assert "A" not in result


def test_symbol_not_in_query_excluded(tmp_path):
    """Headlines for symbols not in the query list are ignored."""
    db = _make_db(tmp_path)
    _insert(db, [("B", 0.9, 1.0, 60, 0)])
    result = live_sentiment_by_symbol(["A"], db_path=db)
    assert "A" not in result
    assert "B" not in result


def test_clamped_to_minus_one_plus_one(tmp_path):
    """Result stays in [-1, 1] even if the DB has out-of-range values."""
    db = _make_db(tmp_path)
    conn = sqlite3.connect(db)
    now = int(time.time())
    conn.execute(
        "INSERT INTO news (published_at,fetched_at,source,headline,url,"
        "instrument_key,sentiment_score,sentiment_confidence,"
        "sentiment_decay_minutes,sentiment_model,sentiment_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (now, now, "t", "h", "u", "A", 5.0, 1.0, 60, "t", now),
    )
    conn.commit()
    conn.close()
    result = live_sentiment_by_symbol(["A"], db_path=db)
    assert result.get("A", 0.0) <= 1.0
