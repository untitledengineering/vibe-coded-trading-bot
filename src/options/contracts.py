"""Options contract data layer.

Two surfaces:
    1. Live options chain — fetched on-demand from Upstox via OptionsApi
    2. Local cache — option_contracts table, populated from the chain or from
       expired-contracts API for backtest

Underlyings supported in v1: NIFTY 50 and BANKNIFTY (the deepest-liquidity
weekly chains for retail credit spreads).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import aiosqlite

from src.utils.config import DB_PATH


# Underlyings we track in v1. Both have weekly expiries on Thursday (NIFTY)
# and historically also weekly on Wednesday (BANKNIFTY) — Upstox returns
# whatever the exchange currently lists.
SUPPORTED_UNDERLYINGS = [
    {
        "instrument_key": "NSE_INDEX|Nifty 50",
        "underlying_symbol": "NIFTY",
        "lot_size_hint": 50,           # actual lot size comes from each contract row
    },
    {
        "instrument_key": "NSE_INDEX|Nifty Bank",
        "underlying_symbol": "BANKNIFTY",
        "lot_size_hint": 15,
    },
]


@dataclass(frozen=True)
class OptionContract:
    instrument_key: str
    underlying: str
    underlying_symbol: str
    expiry_ts: int               # epoch seconds
    expiry_date: str             # 'YYYY-MM-DD'
    strike_price: float
    instrument_type: str         # 'CE' or 'PE'
    lot_size: int


# ---------- DB persistence ----------

async def upsert_contracts(
    contracts: List[OptionContract],
    db_path: Optional[str] = None,
) -> int:
    """Insert-or-update contracts. Returns the count of rows touched."""
    if not contracts:
        return 0
    now = int(time.time())
    rows = [
        (
            c.instrument_key, c.underlying, c.underlying_symbol,
            c.expiry_ts, c.expiry_date, c.strike_price, c.instrument_type,
            c.lot_size, now, 0,
        )
        for c in contracts
    ]
    async with aiosqlite.connect(db_path or DB_PATH) as db:
        await db.executemany(
            """
            INSERT INTO option_contracts
                (instrument_key, underlying, underlying_symbol,
                 expiry_ts, expiry_date, strike_price, instrument_type,
                 lot_size, last_seen_ts, is_expired)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(instrument_key) DO UPDATE SET
                last_seen_ts = excluded.last_seen_ts,
                is_expired   = excluded.is_expired
            """,
            rows,
        )
        await db.commit()
    return len(rows)


async def mark_expired(
    instrument_keys: List[str],
    db_path: Optional[str] = None,
) -> None:
    if not instrument_keys:
        return
    placeholders = ",".join("?" * len(instrument_keys))
    async with aiosqlite.connect(db_path or DB_PATH) as db:
        await db.execute(
            f"UPDATE option_contracts SET is_expired = 1 WHERE instrument_key IN ({placeholders})",
            tuple(instrument_keys),
        )
        await db.commit()


async def list_contracts_for_expiry(
    underlying: str,
    expiry_date: str,
    db_path: Optional[str] = None,
) -> List[OptionContract]:
    async with aiosqlite.connect(db_path or DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT * FROM option_contracts
            WHERE underlying = ? AND expiry_date = ?
            ORDER BY instrument_type, strike_price
            """,
            (underlying, expiry_date),
        ) as cur:
            return [_row_to_contract(r) for r in await cur.fetchall()]


def _row_to_contract(row) -> OptionContract:
    return OptionContract(
        instrument_key=row["instrument_key"],
        underlying=row["underlying"],
        underlying_symbol=row["underlying_symbol"],
        expiry_ts=row["expiry_ts"],
        expiry_date=row["expiry_date"],
        strike_price=row["strike_price"],
        instrument_type=row["instrument_type"],
        lot_size=row["lot_size"],
    )


# ---------- Live chain fetch (Upstox SDK adapter, will be exercised tomorrow) ----------

def _parse_expiry_iso(expiry_field) -> tuple[int, str]:
    """Upstox returns expiry as an ISO datetime or epoch ms; normalise to both."""
    if isinstance(expiry_field, (int, float)):
        # Epoch ms vs s heuristic.
        if expiry_field > 10**12:
            ts = int(expiry_field // 1000)
        else:
            ts = int(expiry_field)
        from datetime import datetime
        return ts, datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    if isinstance(expiry_field, str):
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(expiry_field.replace("Z", "+00:00"))
        except ValueError:
            dt = datetime.strptime(expiry_field[:10], "%Y-%m-%d")
        return int(dt.timestamp()), dt.strftime("%Y-%m-%d")
    raise ValueError(f"Unknown expiry format: {expiry_field!r}")


def parse_option_contracts_response(api_response, underlying: str, underlying_symbol: str) -> List[OptionContract]:
    """Convert Upstox SDK's OptionsApi.get_option_contracts response into our model.

    The SDK returns a list of GetOptionContractResponse items, each with
    .instrument_key, .strike_price, .instrument_type, .expiry, .lot_size etc.
    """
    contracts: List[OptionContract] = []
    data = getattr(api_response, "data", None) or api_response
    for item in data:
        d = item.to_dict() if hasattr(item, "to_dict") else dict(item)
        try:
            ts, date_str = _parse_expiry_iso(d.get("expiry"))
            contracts.append(OptionContract(
                instrument_key=str(d["instrument_key"]),
                underlying=underlying,
                underlying_symbol=underlying_symbol,
                expiry_ts=ts,
                expiry_date=date_str,
                strike_price=float(d["strike_price"]),
                instrument_type=str(d["instrument_type"]),
                lot_size=int(d.get("lot_size") or d.get("minimum_lot") or 0),
            ))
        except (KeyError, ValueError, TypeError):
            continue
    return contracts
