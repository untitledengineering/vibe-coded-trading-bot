"""Tests for the paper executor — fills, slippage, P&L accounting, persistence."""

import aiosqlite
import pytest

from src.bot.engine import EngineConfig, OrderIntent
from src.paper.executor import PaperExecutor
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
        await d.commit()
    mocker.patch("src.paper.persistence.DB_PATH", db_path)
    return db_path


@pytest.mark.asyncio
async def test_fill_long_applies_slippage_up(isolated_db):
    ex = PaperExecutor(config=EngineConfig(), slippage_bps=5.0, model_name="v1")
    intent = OrderIntent(
        instrument_key="NSE_EQ|X", side="long", qty=10,
        reason="long_top_pick", predicted_return=0.005,
    )
    pos = await ex.fill_intent(intent, fill_ts=1_000_000, last_quote_price=100.0)
    # 5 bps above 100 -> 100.05
    assert pos.entry_price == pytest.approx(100.05)
    # SL/TP off the slipped entry, matching backtest convention.
    assert pos.stop_loss_price == pytest.approx(100.05 * (1 - 0.005))
    assert pos.target_price == pytest.approx(100.05 * (1 + 0.008))


@pytest.mark.asyncio
async def test_fill_short_applies_slippage_down(isolated_db):
    ex = PaperExecutor(config=EngineConfig(), slippage_bps=5.0, model_name="v1")
    intent = OrderIntent(
        instrument_key="NSE_EQ|X", side="short", qty=10,
        reason="short_bottom_pick", predicted_return=-0.005,
    )
    pos = await ex.fill_intent(intent, fill_ts=1_000_000, last_quote_price=100.0)
    assert pos.entry_price == pytest.approx(99.95)


@pytest.mark.asyncio
async def test_fill_persists_to_db_and_close_finalises(isolated_db):
    ex = PaperExecutor(config=EngineConfig(), slippage_bps=5.0, model_name="v1")
    intent = OrderIntent(
        instrument_key="NSE_EQ|X", side="long", qty=100,
        reason="long_top_pick", predicted_return=0.005,
    )
    pos = await ex.fill_intent(intent, fill_ts=1_000_000, last_quote_price=100.0)
    open_now = await P.list_open_positions()
    assert len(open_now) == 1
    assert open_now[0].is_open

    await ex.close_position(pos, exit_ts=1_000_300, exit_price=102.0, reason="target")
    assert await P.list_open_positions() == []
    # Realised P&L should reflect gross gain (~₹195) less round-trip cost (~₹40).
    assert pos.realised_pnl_inr is not None
    assert pos.realised_pnl_inr > 100  # net positive on a +2% move
    assert pos.realised_pnl_inr < 200
