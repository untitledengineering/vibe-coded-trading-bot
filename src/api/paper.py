"""HTTP surface for the paper-trading engine.

GET  /paper/status       — current snapshot (open positions, P&L, halt state, etc.)
POST /paper/halt         — manually halt new entries
POST /paper/resume       — clear a halt
POST /paper/report       — force EOD report generation (mostly useful for testing)
GET  /paper/scanner      — top gainers/losers from bars_live (?window=15)
GET  /paper/signals      — cached model predictions from the last decision cycle
GET  /paper/news         — recent scored news headlines (last 6h)
POST /paper/force-entry  — manually place a paper trade from the UI
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
import aiosqlite
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from typing import Optional

from src.bot.engine import OrderIntent
from src.db.database import get_valid_token
from src.paper import loop as paper_loop
from src.paper.persistence import list_open_positions, set_halt, get_halt_state
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


@router.get("/paper/quotes")
async def paper_quotes():
    """Fresh LTP for every open position. Polled every 2s by the UI.
    Uses stream quotes if available; falls back to Upstox REST OHLC for the rest."""
    positions = await list_open_positions()
    if not positions:
        return {}

    from src.api import streamer_manager
    result: dict[str, float] = {}
    missing = []

    for pos in positions:
        q = streamer_manager.last_quote_by_symbol.get(pos.instrument_key)
        if q is not None:
            result[pos.instrument_key] = q
        else:
            missing.append(pos.instrument_key)

    # No REST fallback here — this endpoint is polled every 1s and a blocking
    # Upstox call would freeze the server. The backend _refresh_rest_quotes runs
    # every 30s in the mark-to-market loop to fill gaps for illiquid stocks.
    return result


@router.post("/paper/close/{instrument_key}")
async def paper_close_position(instrument_key: str):
    """Manually close a specific open position at the current LTP."""
    engine = paper_loop.get_engine()
    if engine is None or engine.executor is None:
        return JSONResponse(status_code=503, content={"error": "engine not running"})
    positions = await list_open_positions()
    pos = next((p for p in positions if p.instrument_key == instrument_key), None)
    if pos is None:
        return JSONResponse(status_code=404, content={"error": "position not found"})
    from src.api import streamer_manager
    quote: Optional[float] = streamer_manager.last_quote_by_symbol.get(instrument_key)

    if quote is None:
        # Fall back to REST OHLC so manual close works even when stream has no quote yet.
        token = await get_valid_token()
        if token:
            try:
                import upstox_client

                def _ohlc_ltp(tok: str, key: str) -> Optional[float]:
                    cfg = upstox_client.Configuration()
                    cfg.access_token = tok
                    api = upstox_client.MarketQuoteApi(upstox_client.ApiClient(cfg))
                    resp = api.get_market_quote_ohlc(symbol=key, interval="1d", api_version="2.0")
                    if resp and hasattr(resp, "data") and resp.data:
                        for q in resp.data.values():
                            lp = q.last_price if hasattr(q, "last_price") else None
                            if lp:
                                return float(lp)
                    return None

                loop = asyncio.get_running_loop()
                quote = await loop.run_in_executor(None, _ohlc_ltp, token, instrument_key)
            except Exception as exc:
                logger.warning(f"manual-close: OHLC fallback failed for {instrument_key}: {exc}")

    if quote is None:
        return JSONResponse(status_code=503, content={"error": "no live quote — try again in a second"})

    await engine.executor.close_position(pos, exit_ts=int(time.time()), exit_price=quote, reason="manual_close")
    logger.info(f"Manual close: {instrument_key} @ {quote}")
    return {"ok": True, "instrument_key": instrument_key, "exit_price": quote}


@router.post("/paper/extend-loss-cap")
async def paper_extend_loss_cap(amount: float = Query(500.0, ge=100.0, le=5000.0)):
    """Extend the daily loss cap by `amount` INR and clear any loss-cap halt."""
    engine = paper_loop.get_engine()
    if engine is None:
        return JSONResponse(status_code=503, content={"error": "engine not running"})
    engine.loss_cap_extension += amount
    # Always clear any active halt when user explicitly extends the cap.
    halt = await get_halt_state(time.time())
    if halt["halted"]:
        await set_halt(reason="cap_extended_by_user", halted=False, now_ts=time.time())
        logger.info(f"Loss cap extended +₹{amount:.0f} and halt cleared (total extension ₹{engine.loss_cap_extension:.0f})")
    else:
        logger.info(f"Loss cap extended +₹{amount:.0f} (total extension ₹{engine.loss_cap_extension:.0f})")
    return {
        "ok": True,
        "extended_by": amount,
        "total_extension_inr": engine.loss_cap_extension,
        "effective_cap_inr": engine.daily_loss_cap_inr + engine.loss_cap_extension,
    }


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
        # Use the bar at/before target_past_ts as reference. During the first
        # `window` minutes of the session no such bar exists, so fall back to
        # the oldest available bar and label it "since open".
        past_bars = [(ts, c) for ts, c in bars if ts <= target_past_ts]
        if past_bars:
            close_prev = past_bars[-1][1]
            elapsed_minutes = window
        else:
            close_prev = bars[0][1]
            elapsed_minutes = max(1, round((bars[-1][0] - bars[0][0]) / 60))
        if not close_prev or close_prev <= 0:
            continue
        ret_pct = (close_now - close_prev) / close_prev * 100
        results.append({
            "instrument_key": key,
            "trading_symbol": symbols.get(key, key),
            "close_now": round(close_now, 2),
            "close_prev": round(close_prev, 2),
            "return_pct": round(ret_pct, 3),
            "elapsed_minutes": elapsed_minutes,
            "ltp": streamer_manager.last_quote_by_symbol.get(key),
        })

    results.sort(key=lambda x: x["return_pct"], reverse=True)
    # Separate by sign so the same symbol can't appear in both lists.
    gainers = [r for r in results if r["return_pct"] > 0][:30]
    losers  = [r for r in reversed(results) if r["return_pct"] < 0][:30]

    # Annotate with model signal if available
    engine = paper_loop.get_engine()
    if engine and engine.last_signals:
        for item in gainers + losers:
            item["model_signal"] = engine.last_signals.get(item["instrument_key"])
            item["sentiment_score"] = engine.last_sentiment_scores.get(item["instrument_key"])

    # Flag early-session: most symbols fell back to "since open" reference
    early_session = sum(1 for r in results if r["elapsed_minutes"] < window) > len(results) // 2

    return {
        "window_minutes": window,
        "early_session": early_session,
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


@router.post("/paper/force-entry")
async def paper_force_entry(
    instrument_key: str,
    side: str,
    qty: Optional[int] = None,
):
    """Manually place a paper trade from the UI.

    Resolves LTP from the live streamer first; falls back to the OHLC API.
    qty defaults to floor(capital / ltp) if not provided.
    """
    if side not in ("long", "short"):
        return JSONResponse(status_code=400, content={"error": "side must be 'long' or 'short'"})

    engine = paper_loop.get_engine()
    if engine is None or engine.executor is None:
        return JSONResponse(status_code=503, content={"error": "paper engine not running"})

    from src.api import streamer_manager

    ltp: Optional[float] = streamer_manager.last_quote_by_symbol.get(instrument_key)

    if ltp is None:
        # Fall back: fetch via OHLC API (same approach as market movers, single symbol)
        token = await get_valid_token()
        if token:
            try:
                import upstox_client

                def _fetch_single(tok: str, key: str) -> Optional[float]:
                    cfg = upstox_client.Configuration()
                    cfg.access_token = tok
                    api = upstox_client.MarketQuoteApi(upstox_client.ApiClient(cfg))
                    # Upstox OHLC API takes pipe-format keys (NSE_EQ|ISIN)
                    resp = api.get_market_quote_ohlc(
                        symbol=key, interval="1d", api_version="2.0"
                    )
                    if resp and hasattr(resp, "data") and resp.data:
                        for q in resp.data.values():
                            lp = q.last_price if hasattr(q, "last_price") else None
                            if lp:
                                return float(lp)
                    return None

                loop = asyncio.get_running_loop()
                ltp = await loop.run_in_executor(None, _fetch_single, token, instrument_key)
            except Exception as exc:
                logger.warning(f"force-entry: OHLC fallback failed for {instrument_key}: {exc}")

    if ltp is None:
        return JSONResponse(
            status_code=503,
            content={"error": "no live quote available — try again in a moment"},
        )

    if qty is None:
        qty = max(1, int(engine.config.capital_inr // ltp))

    intent = OrderIntent(
        instrument_key=instrument_key,
        side=side,
        qty=qty,
        reason=f"manual_{side}",
        predicted_return=0.0,
    )

    try:
        await engine.executor.fill_intent(
            intent,
            fill_ts=int(time.time()),
            last_quote_price=ltp,
            entry_sentiment_score=None,
        )
    except Exception as exc:
        logger.error(f"force-entry: fill_intent failed: {exc}")
        return JSONResponse(status_code=500, content={"error": str(exc)})

    logger.info(f"force-entry: {side} {qty} {instrument_key} @ {ltp:.2f}")
    return {
        "ok": True,
        "instrument_key": instrument_key,
        "side": side,
        "qty": qty,
        "entry_price": ltp,
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
