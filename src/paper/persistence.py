"""SQLite IO for paper-trading state.

Holds the source of truth: every paper position lands in `paper_positions`
the moment it opens. The in-memory cache in loop.py is just a derived view —
a bot crash mid-day means we rehydrate from this table on restart and keep
managing the open positions.
"""

from __future__ import annotations

import time
from typing import List, Optional

import aiosqlite

from src.bot.positions import Position
from src.features.technical import IST_OFFSET_SECONDS, SECONDS_PER_DAY
from src.utils.config import DB_PATH


def ist_date_str(ts: Optional[float] = None) -> str:
    """YYYY-MM-DD in IST for the given timestamp (now if None)."""
    now = ts if ts is not None else time.time()
    ist = time.gmtime(now + IST_OFFSET_SECONDS)
    return time.strftime("%Y-%m-%d", ist)


def ist_session_date(ts: float) -> int:
    """Integer 'IST days since epoch' — matches engine._ist_session_date()."""
    return int((ts + IST_OFFSET_SECONDS) // SECONDS_PER_DAY)


# ---------- Position persistence ----------

async def insert_open_position(
    pos: Position,
    predicted_return: Optional[float],
    model_name: Optional[str],
    entry_sentiment_score: Optional[float] = None,
    db_path: Optional[str] = None,
) -> int:
    """Insert a fresh open position. Returns the row id."""
    async with aiosqlite.connect(db_path or DB_PATH) as db:
        cur = await db.execute(
            """
            INSERT INTO paper_positions
                (instrument_key, side, qty, entry_ts, entry_price,
                 stop_loss_price, target_price,
                 predicted_return, model_name, entry_sentiment_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pos.instrument_key, pos.side, pos.qty, pos.entry_ts, pos.entry_price,
                pos.stop_loss_price, pos.target_price,
                predicted_return, model_name, entry_sentiment_score,
            ),
        )
        await db.commit()
        return cur.lastrowid


async def mark_position_closed(
    pos: Position,
    db_path: Optional[str] = None,
) -> None:
    """Finalise a position: write exit_ts/price/reason/pnl onto its row."""
    async with aiosqlite.connect(db_path or DB_PATH) as db:
        await db.execute(
            """
            UPDATE paper_positions
            SET exit_ts = ?, exit_price = ?, exit_reason = ?, realised_pnl_inr = ?
            WHERE instrument_key = ? AND entry_ts = ?
            """,
            (
                pos.exit_ts, pos.exit_price, pos.exit_reason, pos.realised_pnl_inr,
                pos.instrument_key, pos.entry_ts,
            ),
        )
        await db.commit()


def _row_to_position(row) -> Position:
    keys = row.keys() if hasattr(row, "keys") else []
    return Position(
        instrument_key=row["instrument_key"],
        side=row["side"],
        qty=row["qty"],
        entry_ts=row["entry_ts"],
        entry_price=row["entry_price"],
        stop_loss_price=row["stop_loss_price"],
        target_price=row["target_price"],
        exit_ts=row["exit_ts"],
        exit_price=row["exit_price"],
        exit_reason=row["exit_reason"],
        realised_pnl_inr=row["realised_pnl_inr"],
        predicted_return=row["predicted_return"] if "predicted_return" in keys else None,
        model_name=row["model_name"] if "model_name" in keys else None,
        entry_sentiment_score=row["entry_sentiment_score"] if "entry_sentiment_score" in keys else None,
    )


async def list_open_positions(db_path: Optional[str] = None) -> List[Position]:
    async with aiosqlite.connect(db_path or DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM paper_positions WHERE exit_ts IS NULL ORDER BY entry_ts"
        ) as cur:
            return [_row_to_position(r) for r in await cur.fetchall()]


async def list_closed_positions_today(
    now_ts: float,
    db_path: Optional[str] = None,
) -> List[Position]:
    today = ist_session_date(now_ts)
    today_start_utc = today * SECONDS_PER_DAY - IST_OFFSET_SECONDS
    today_end_utc = today_start_utc + SECONDS_PER_DAY
    async with aiosqlite.connect(db_path or DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT * FROM paper_positions
            WHERE exit_ts IS NOT NULL
              AND entry_ts >= ? AND entry_ts < ?
            ORDER BY entry_ts
            """,
            (today_start_utc, today_end_utc),
        ) as cur:
            return [_row_to_position(r) for r in await cur.fetchall()]


async def list_positions_today(
    now_ts: float,
    db_path: Optional[str] = None,
) -> List[Position]:
    """All positions (open + closed) entered today in IST."""
    today = ist_session_date(now_ts)
    today_start_utc = today * SECONDS_PER_DAY - IST_OFFSET_SECONDS
    today_end_utc = today_start_utc + SECONDS_PER_DAY
    async with aiosqlite.connect(db_path or DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT * FROM paper_positions
            WHERE entry_ts >= ? AND entry_ts < ?
            ORDER BY entry_ts
            """,
            (today_start_utc, today_end_utc),
        ) as cur:
            return [_row_to_position(r) for r in await cur.fetchall()]


async def todays_realised_pnl(now_ts: float, db_path: Optional[str] = None) -> float:
    """Sum of realised P&L (after costs) for trades closed today. 0.0 if none."""
    today = ist_session_date(now_ts)
    today_start_utc = today * SECONDS_PER_DAY - IST_OFFSET_SECONDS
    today_end_utc = today_start_utc + SECONDS_PER_DAY
    async with aiosqlite.connect(db_path or DB_PATH) as db:
        async with db.execute(
            """
            SELECT COALESCE(SUM(realised_pnl_inr), 0.0)
            FROM paper_positions
            WHERE exit_ts IS NOT NULL
              AND entry_ts >= ? AND entry_ts < ?
            """,
            (today_start_utc, today_end_utc),
        ) as cur:
            row = await cur.fetchone()
            return float(row[0]) if row else 0.0


# ---------- Halt state ----------

async def set_halt(
    reason: str,
    halted: bool,
    now_ts: Optional[float] = None,
    db_path: Optional[str] = None,
) -> None:
    """Idempotent insert-or-update of today's halt state."""
    ts = now_ts if now_ts is not None else time.time()
    date = ist_date_str(ts)
    halt_int = 1 if halted else 0
    halted_at = int(ts) if halted else None
    async with aiosqlite.connect(db_path or DB_PATH) as db:
        # INSERT OR IGNORE then UPDATE — keeps started_at unchanged.
        await db.execute(
            """
            INSERT OR IGNORE INTO paper_session_state
                (session_date, halted, halt_reason, halted_at, started_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (date, halt_int, reason if halted else None, halted_at, int(ts)),
        )
        await db.execute(
            """
            UPDATE paper_session_state
            SET halted = ?, halt_reason = ?, halted_at = ?
            WHERE session_date = ?
            """,
            (halt_int, reason if halted else None, halted_at, date),
        )
        await db.commit()


async def get_halt_state(
    now_ts: Optional[float] = None,
    db_path: Optional[str] = None,
) -> dict:
    ts = now_ts if now_ts is not None else time.time()
    date = ist_date_str(ts)
    async with aiosqlite.connect(db_path or DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM paper_session_state WHERE session_date = ?",
            (date,),
        ) as cur:
            row = await cur.fetchone()
            if not row:
                return {
                    "session_date": date,
                    "halted": False,
                    "halt_reason": None,
                    "halted_at": None,
                    "started_at": None,
                }
            return {
                "session_date": row["session_date"],
                "halted": bool(row["halted"]),
                "halt_reason": row["halt_reason"],
                "halted_at": row["halted_at"],
                "started_at": row["started_at"],
            }
