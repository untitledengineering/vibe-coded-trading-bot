"""5-min bar aggregator tests."""

import pandas as pd
import pytest

from src.strategy.bars5m import aggregate_to_5m


def _bar(ts, o, h, l, c, v):
    return {"minute_ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def test_empty_input_returns_empty():
    out = aggregate_to_5m(pd.DataFrame(columns=["minute_ts", "open", "high", "low", "close", "volume"]))
    assert out.empty


def test_five_one_minute_bars_collapse_into_one_5m_bar():
    bars = pd.DataFrame([
        _bar(0,   100, 102, 99,  101, 1000),
        _bar(60,  101, 103, 100, 102, 800),
        _bar(120, 102, 104, 101, 103, 900),
        _bar(180, 103, 105, 102, 104, 700),
        _bar(240, 104, 106, 103, 105, 600),
    ])
    out = aggregate_to_5m(bars)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["minute_ts"] == 0
    assert row["open"] == 100
    assert row["high"] == 106
    assert row["low"] == 99
    assert row["close"] == 105
    assert row["volume"] == 4000


def test_bars_split_across_two_5m_buckets():
    bars = pd.DataFrame([
        _bar(0,   100, 101, 99,  100, 100),
        _bar(60,  100, 102, 99,  101, 100),
        _bar(300, 101, 103, 100, 102, 100),  # next bucket starts at 300
        _bar(360, 102, 104, 101, 103, 100),
    ])
    out = aggregate_to_5m(bars).sort_values("minute_ts").reset_index(drop=True)
    assert len(out) == 2
    assert list(out["minute_ts"]) == [0, 300]
    assert out.iloc[1]["open"] == 101
    assert out.iloc[1]["close"] == 103


def test_aggregator_handles_gaps_inside_bucket():
    bars = pd.DataFrame([
        _bar(0,   100, 101, 99,  100, 100),
        # 60s and 120s missing (data gap)
        _bar(180, 99,  100, 98,  99,  100),
        _bar(240, 99,  101, 98,  100, 100),
    ])
    out = aggregate_to_5m(bars)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["open"] == 100
    assert row["high"] == 101
    assert row["low"] == 98
    assert row["close"] == 100
