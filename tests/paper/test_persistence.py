"""SQLite CRUD tests for the paper-trading persistence layer."""

import time

import aiosqlite
import pytest

from src.bot.positions import Position
from src.paper import persistence as P


@pytest.fixture
async def db(tmp_path):
    db_path = str(tmp_path / "paper.db")
    async with aiosqlite.connect(db_path) as d:
        await d.execute("""
            CREATE TABLE paper_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instrument_key TEXT NOT NULL,
                side TEXT NOT NULL,
                qty INTEGER NOT NULL,
                entry_ts INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss_price REAL NOT NULL,
                target_price REAL NOT NULL,
                exit_ts INTEGER,
                exit_price REAL,
                exit_reason TEXT,
                realised_pnl_inr REAL,
                predicted_return REAL,
                model_name TEXT,
                entry_sentiment_score REAL,
                UNIQUE(instrument_key, entry_ts)
            )
        """)
        await d.execute("""
            CREATE TABLE paper_session_state (
                session_date TEXT PRIMARY KEY,
                halted INTEGER NOT NULL DEFAULT 0,
                halt_reason TEXT,
                halted_at INTEGER,
                started_at INTEGER
            )
        """)
        await d.commit()
    return db_path


def _pos(key="NSE_EQ|TEST", entry_ts=1_000_000, side="long"):
    return Position(
        instrument_key=key, side=side, qty=10, entry_ts=entry_ts,
        entry_price=100.0, stop_loss_price=99.5, target_price=100.8,
    )


@pytest.mark.asyncio
async def test_insert_and_list_open(db):
    await P.insert_open_position(_pos(), predicted_return=0.005, model_name="v1", db_path=db)
    open_now = await P.list_open_positions(db_path=db)
    assert len(open_now) == 1
    assert open_now[0].instrument_key == "NSE_EQ|TEST"
    assert open_now[0].is_open


@pytest.mark.asyncio
async def test_mark_closed_moves_position_out_of_open(db):
    p = _pos()
    await P.insert_open_position(p, predicted_return=0.005, model_name="v1", db_path=db)
    p.close(exit_ts=1_000_300, exit_price=100.8, reason="target", costs_inr=20.0)
    await P.mark_position_closed(p, db_path=db)
    assert await P.list_open_positions(db_path=db) == []


@pytest.mark.asyncio
async def test_todays_realised_pnl_sums_only_closed_today(db):
    # Two trades today, both closed
    now = time.time()
    today_entry = int(now) - 600
    p1 = _pos(key="NSE_EQ|A", entry_ts=today_entry)
    p2 = _pos(key="NSE_EQ|B", entry_ts=today_entry + 10)
    await P.insert_open_position(p1, 0.0, "v1", db_path=db)
    await P.insert_open_position(p2, 0.0, "v1", db_path=db)
    p1.close(exit_ts=today_entry + 60, exit_price=101.0, reason="target", costs_inr=20.0)
    p2.close(exit_ts=today_entry + 60, exit_price=99.0,  reason="stop_loss", costs_inr=20.0)
    await P.mark_position_closed(p1, db_path=db)
    await P.mark_position_closed(p2, db_path=db)

    pnl = await P.todays_realised_pnl(now, db_path=db)
    # p1: gross +10, costs 20 -> -10. p2: gross -10, costs 20 -> -30. Sum -40.
    assert pnl == pytest.approx(-40.0)


@pytest.mark.asyncio
async def test_halt_state_roundtrip(db):
    now = time.time()
    initial = await P.get_halt_state(now, db_path=db)
    assert initial["halted"] is False
    assert initial["halt_reason"] is None

    await P.set_halt(reason="daily_loss_cap", halted=True, now_ts=now, db_path=db)
    after = await P.get_halt_state(now, db_path=db)
    assert after["halted"] is True
    assert after["halt_reason"] == "daily_loss_cap"
    assert after["halted_at"] is not None

    await P.set_halt(reason="manual_resume", halted=False, now_ts=now, db_path=db)
    cleared = await P.get_halt_state(now, db_path=db)
    assert cleared["halted"] is False


@pytest.mark.asyncio
async def test_double_open_same_symbol_same_ts_blocked(db):
    """UNIQUE(instrument_key, entry_ts) protects against accidental retries."""
    p = _pos()
    await P.insert_open_position(p, 0.0, "v1", db_path=db)
    with pytest.raises(Exception):  # IntegrityError or aiosqlite wrapper
        await P.insert_open_position(p, 0.0, "v1", db_path=db)
