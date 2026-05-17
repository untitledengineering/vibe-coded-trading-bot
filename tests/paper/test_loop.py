"""Engine-loop tests. We exercise `_cycle_once` with mocked feature data and
mocked LTPs so we can drive precise scenarios — SL hits, EOD exits, halts."""

import time
from unittest.mock import MagicMock

import aiosqlite
import numpy as np
import pandas as pd
import pytest

from src.api import streamer_manager
from src.bot.engine import EngineConfig
from src.features.technical import FEATURE_COLUMNS
from src.paper import loop as paper_loop
from src.paper import persistence as P


@pytest.fixture(autouse=True)
async def isolated_db(tmp_path, mocker):
    db_path = str(tmp_path / "paper.db")
    async with aiosqlite.connect(db_path) as d:
        await d.execute("""
            CREATE TABLE paper_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instrument_key TEXT, side TEXT, qty INTEGER,
                entry_ts INTEGER, entry_price REAL,
                stop_loss_price REAL, target_price REAL,
                exit_ts INTEGER, exit_price REAL, exit_reason TEXT,
                realised_pnl_inr REAL, predicted_return REAL, model_name TEXT,
                entry_sentiment_score REAL,
                UNIQUE(instrument_key, entry_ts)
            )
        """)
        await d.execute("""
            CREATE TABLE paper_session_state (
                session_date TEXT PRIMARY KEY, halted INTEGER NOT NULL DEFAULT 0,
                halt_reason TEXT, halted_at INTEGER, started_at INTEGER
            )
        """)
        await d.commit()
    mocker.patch("src.paper.persistence.DB_PATH", db_path)
    yield db_path


@pytest.fixture(autouse=True)
def market_open(monkeypatch):
    """Force is_market_open() True so we don't care about wall-clock."""
    monkeypatch.setattr(streamer_manager, "is_market_open", lambda *a, **kw: True)


@pytest.fixture
def features_one_row():
    row = {"instrument_key": "NSE_EQ|X", "close": 100.0}
    for c in FEATURE_COLUMNS:
        row[c] = 0.0
    return pd.DataFrame([row])


def _engine_with_model(monkeypatch, predictions):
    """Build a PaperEngine wired to a fake model that returns `predictions`."""
    eng = paper_loop.PaperEngine(
        config=EngineConfig(
            min_predicted_edge=0.001,
            top_k_long=1, top_k_short=0,
            entry_window_open_minute_ist=0, entry_window_close_minute_ist=23 * 60 + 59,
            forced_exit_minute_ist=23 * 60 + 58,  # effectively never in tests below
            cooldown_minutes=0, max_trades_per_symbol_per_day=99,
        ),
        interval_seconds=0,
    )
    eng.model = MagicMock()
    eng.model.feature_columns = list(FEATURE_COLUMNS)
    eng.executor = paper_loop.PaperExecutor(config=eng.config, model_name="test")
    monkeypatch.setattr(paper_loop, "decide", _make_decide_stub(predictions))
    return eng


def _make_decide_stub(predictions):
    """Build a decide() stand-in that calls the real decide with our score patch."""
    from src.bot import engine as engine_mod
    real_decide = engine_mod.decide

    def stub(*args, **kwargs):
        # Patch score for this call only.
        from src.bot import engine as e
        e.score = lambda model, frame: np.array(predictions)
        return real_decide(*args, **kwargs)
    return stub


@pytest.mark.asyncio
async def test_cycle_with_signal_queues_intent_and_fills_next_cycle(
    monkeypatch, features_one_row
):
    eng = _engine_with_model(monkeypatch, [0.01])
    monkeypatch.setattr(paper_loop, "build_live_feature_frame", lambda keys: features_one_row)
    streamer_manager.last_quote_by_symbol.clear()
    streamer_manager.last_quote_by_symbol["NSE_EQ|X"] = 100.0

    # First cycle: minute_now is fresh, decide() runs, intent queued.
    await eng._cycle_once()
    assert len(eng._pending_intents) == 1

    # Second cycle in the SAME minute: decide() doesn't run again, but pending fills.
    # Bump time forward by 1s but stay in the same minute.
    await eng._cycle_once()
    open_now = await P.list_open_positions()
    assert len(open_now) == 1
    assert open_now[0].instrument_key == "NSE_EQ|X"
    assert eng._pending_intents == []


@pytest.mark.asyncio
async def test_open_position_closed_on_stop_loss(monkeypatch, features_one_row):
    eng = _engine_with_model(monkeypatch, [0.0])
    monkeypatch.setattr(paper_loop, "build_live_feature_frame", lambda keys: features_one_row)
    streamer_manager.last_quote_by_symbol.clear()

    # Seed an open position manually so we can test the SL exit branch.
    from src.bot.positions import Position
    pos = Position(
        instrument_key="NSE_EQ|X", side="long", qty=10,
        entry_ts=int(time.time()) - 60,
        entry_price=100.0, stop_loss_price=99.5, target_price=100.8,
    )
    await P.insert_open_position(pos, 0.0, "test")

    # Quote dropped below SL.
    streamer_manager.last_quote_by_symbol["NSE_EQ|X"] = 99.0

    await eng._cycle_once()
    assert await P.list_open_positions() == []


@pytest.mark.asyncio
async def test_daily_loss_cap_triggers_halt(monkeypatch, features_one_row):
    eng = _engine_with_model(monkeypatch, [0.0])
    eng.daily_loss_cap_inr = 50.0  # tight cap for the test
    monkeypatch.setattr(paper_loop, "build_live_feature_frame", lambda keys: features_one_row)
    streamer_manager.last_quote_by_symbol.clear()

    # Insert a closed losing trade well past the cap.
    # qty=100 × (-2.0) drop − 20 costs = realised -220, beats the 50 cap easily.
    from src.bot.positions import Position
    pos = Position(
        instrument_key="NSE_EQ|LOSER", side="long", qty=100,
        entry_ts=int(time.time()) - 600,
        entry_price=100.0, stop_loss_price=98.0, target_price=102.0,
    )
    await P.insert_open_position(pos, 0.0, "test")
    pos.close(exit_ts=int(time.time()) - 300, exit_price=98.0, reason="stop_loss", costs_inr=20.0)
    await P.mark_position_closed(pos)

    await eng._cycle_once()
    halt = await P.get_halt_state(time.time())
    assert halt["halted"] is True
    assert "daily_loss_cap_breached" in halt["halt_reason"]


@pytest.mark.asyncio
async def test_halt_blocks_new_intents(monkeypatch, features_one_row):
    eng = _engine_with_model(monkeypatch, [0.05])
    monkeypatch.setattr(paper_loop, "build_live_feature_frame", lambda keys: features_one_row)
    streamer_manager.last_quote_by_symbol.clear()
    streamer_manager.last_quote_by_symbol["NSE_EQ|X"] = 100.0

    await P.set_halt(reason="manual", halted=True, now_ts=time.time())
    await eng._cycle_once()
    assert eng._pending_intents == []
    assert await P.list_open_positions() == []


@pytest.mark.asyncio
async def test_status_snapshot_returns_expected_keys(monkeypatch):
    eng = _engine_with_model(monkeypatch, [0.0])
    streamer_manager.last_quote_by_symbol.clear()
    snap = await eng.status()
    for key in (
        "running", "halted", "model", "open_positions",
        "trades_today", "realised_pnl_inr", "unrealised_pnl_inr", "net_pnl_inr",
    ):
        assert key in snap
