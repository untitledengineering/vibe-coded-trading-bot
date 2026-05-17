"""HTTP surface for the paper-trading engine.

GET  /paper/status    — current snapshot (open positions, P&L, halt state, etc.)
POST /paper/halt      — manually halt new entries
POST /paper/resume    — clear a halt
POST /paper/report    — force EOD report generation (mostly useful for testing)
GET  /paper/scanner   — top gainers/losers from bars_live (?window=15)
GET  /paper/signals   — cached model predictions from the last decision cycle
GET  /paper/news      — recent scored news headlines (last 6h)
"""

from __future__ import annotations

import time
from collections import defaultdict
import aiosqlite
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from src.paper import loop as paper_loop
from src.paper.persistence import set_halt
from src.paper.report import generate_report
from src.utils.config import DB_PATH
from src.utils.logger import logger

router = APIRouter()


def _symbol_lookup() -> dict:
    try:
        from src.data.universe import load_universe
        return {u["instrument_key"]: u["trading_symbol"] for u in load_universe()}
    except Exception:
        return {}


@router.get("/paper/status")
async def paper_status():
    engine = paper_loop.get_engine()
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"running": False, "reason": "paper engine not initialised"},
        )
    return await engine.status()


@router.post("/paper/halt")
async def paper_halt():
    """Manually halt new entries for the rest of today. Open positions ride."""
    await set_halt(reason="manual", halted=True, now_ts=time.time())
    logger.warning("Paper engine halted manually via HTTP")
    return {"ok": True, "halted": True}


@router.post("/paper/resume")
async def paper_resume():
    """Clear today's halt. Engine resumes generating intents at the next minute."""
    await set_halt(reason="manual_resume", halted=False, now_ts=time.time())
    logger.info("Paper engine resumed manually via HTTP")
    return {"ok": True, "halted": False}


@router.post("/paper/report")
async def paper_report():
    """Generate today's EOD report on demand. Returns the file path."""
    path = await generate_report()
    logger.info(f"Paper report written to {path}")
    return {"ok": True, "path": str(path)}


@router.get("/paper/scanner")
async def paper_scanner(window: int = Query(15, ge=1, le=120)):
    """Top gainers/losers by return over the last `window` minutes from bars_live.

    Returns up to 20 gainers and 20 losers sorted by absolute return.
    Each row includes the current LTP from the streamer if available.
    """
    from src.api import streamer_manager

    now_ts = int(time.time())
    # Fetch enough history to cover the requested window plus a small buffer.
    cutoff = now_ts - (window + 5) * 60
    symbols = _symbol_lookup()

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT instrument_key, minute_ts, close
            FROM bars_live
            WHERE minute_ts >= ?
            ORDER BY instrument_key, minute_ts ASC
            """,
            (cutoff,),
        ) as cur:
            rows = await cur.fetchall()

    groups: dict = defaultdict(list)
    for row in rows:
        groups[row["instrument_key"]].append((row["minute_ts"], row["close"]))

    results = []
    target_past_ts = now_ts - window * 60
    for key, bars in groups.items():
        if not bars:
            continue
        close_now = bars[-1][1]
        past_bars = [(ts, c) for ts, c in bars if ts <= target_past_ts]
        if not past_bars:
            continue
        close_prev = past_bars[-1][1]
        if not close_prev or close_prev <= 0:
            continue
        ret_pct = (close_now - close_prev) / close_prev * 100
        results.append({
            "instrument_key": key,
            "trading_symbol": symbols.get(key, key),
            "close_now": round(close_now, 2),
            "close_prev": round(close_prev, 2),
            "return_pct": round(ret_pct, 3),
            "ltp": streamer_manager.last_quote_by_symbol.get(key),
        })

    results.sort(key=lambda x: x["return_pct"], reverse=True)
    gainers = results[:20]
    losers = list(reversed(results[-20:])) if len(results) >= 20 else sorted(results, key=lambda x: x["return_pct"])[:20]

    # Annotate with model signal if available
    engine = paper_loop.get_engine()
    if engine and engine.last_signals:
        for item in gainers + losers:
            item["model_signal"] = engine.last_signals.get(item["instrument_key"])
            item["sentiment_score"] = engine.last_sentiment_scores.get(item["instrument_key"])

    return {
        "window_minutes": window,
        "computed_at": now_ts,
        "total_symbols": len(results),
        "gainers": gainers,
        "losers": losers,
    }


@router.get("/paper/signals")
async def paper_signals(top_n: int = Query(40, ge=5, le=209)):
    """Model predictions from the last decision cycle for all scored symbols.

    Returns top_n bullish and top_n bearish by predicted return magnitude.
    """
    engine = paper_loop.get_engine()
    if engine is None or not engine.last_signals:
        return {
            "signals": [],
            "computed_at": None,
            "model": None,
        }

    from src.api import streamer_manager
    symbols = _symbol_lookup()

    all_signals = [
        {
            "instrument_key": k,
            "trading_symbol": symbols.get(k, k),
            "predicted_return": round(v, 5),
            "sentiment_score": engine.last_sentiment_scores.get(k),
            "ltp": streamer_manager.last_quote_by_symbol.get(k),
        }
        for k, v in engine.last_signals.items()
    ]
    all_signals.sort(key=lambda x: x["predicted_return"], reverse=True)

    bullish = all_signals[:top_n]
    bearish = list(reversed(all_signals[-top_n:])) if len(all_signals) >= top_n else sorted(all_signals, key=lambda x: x["predicted_return"])[:top_n]

    return {
        "bullish": bullish,
        "bearish": bearish,
        "total_scored": len(all_signals),
        "computed_at": engine.last_decision_ts or None,
        "model": engine.model_name,
    }


@router.get("/paper/news")
async def paper_news(hours: int = Query(6, ge=1, le=48)):
    """Recent scored news headlines. Sorted by published_at desc."""
    now_ts = int(time.time())
    cutoff = now_ts - hours * 3600
    symbols = _symbol_lookup()

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT id, published_at, fetched_at, source, headline, url,
                   instrument_key, sentiment_score, sentiment_confidence,
                   sentiment_model, sentiment_at
            FROM news
            WHERE published_at >= ?
            ORDER BY published_at DESC
            LIMIT 200
            """,
            (cutoff,),
        ) as cur:
            rows = await cur.fetchall()

    items = []
    for r in rows:
        items.append({
            "id": r["id"],
            "published_at": r["published_at"],
            "source": r["source"],
            "headline": r["headline"],
            "url": r["url"],
            "instrument_key": r["instrument_key"],
            "trading_symbol": symbols.get(r["instrument_key"], r["instrument_key"]) if r["instrument_key"] else None,
            "sentiment_score": r["sentiment_score"],
            "sentiment_confidence": r["sentiment_confidence"],
            "sentiment_model": r["sentiment_model"],
        })

    # Market-wide sentiment: average of scored headlines with no specific instrument
    market_items = [i for i in items if i["instrument_key"] is None and i["sentiment_score"] is not None]
    market_sentiment = (
        round(sum(i["sentiment_score"] for i in market_items) / len(market_items), 3)
        if market_items else None
    )

    return {
        "news": items,
        "total": len(items),
        "hours": hours,
        "computed_at": now_ts,
        "market_sentiment": market_sentiment,
    }
