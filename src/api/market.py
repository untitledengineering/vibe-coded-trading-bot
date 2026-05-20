"""Market-wide movers endpoint.

GET /market/movers?limit=25
  — downloads NSE_EQ universe on first call, caches it
  — batches OHLC queries (500/batch) and runs them concurrently
  — returns top gainers and losers by day % change
  — result is cached 60 s
"""

from __future__ import annotations

import asyncio
import gzip
import io
import json
import time
import urllib.request
from typing import Optional

import upstox_client
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from src.db.database import get_valid_token
from src.utils.logger import logger

router = APIRouter()

# ── Module-level caches ────────────────────────────────────────────────────────
_FULL_UNIVERSE: Optional[list] = None

_MOVERS_CACHE: Optional[dict] = None
_MOVERS_CACHE_AT: float = 0.0

_UNIVERSE_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
_CACHE_TTL_SECONDS = 15
_BATCH_SIZE = 500


def _download_universe() -> list:
    """Download and parse the NSE instrument list (gzipped JSON)."""
    logger.info("Downloading NSE universe from Upstox …")
    with urllib.request.urlopen(_UNIVERSE_URL, timeout=30) as resp:
        raw = resp.read()
    data = json.loads(gzip.decompress(raw))
    # Keep only equity instruments (NSE_EQ segment)
    eq = [r for r in data if r.get("segment") == "NSE_EQ" or
          str(r.get("instrument_key", "")).startswith("NSE_EQ|")]
    logger.info(f"NSE universe loaded: {len(eq)} equity instruments")
    return eq


def _get_universe() -> list:
    global _FULL_UNIVERSE
    if _FULL_UNIVERSE is None:
        _FULL_UNIVERSE = _download_universe()
    return _FULL_UNIVERSE


def _fetch_ohlc_batch(token: str, keys: list[str]) -> dict:
    """Synchronous: fetch OHLC for a batch of instrument keys. Run in executor."""
    cfg = upstox_client.Configuration()
    cfg.access_token = token
    api = upstox_client.MarketQuoteApi(upstox_client.ApiClient(cfg))
    symbol_str = ",".join(keys)
    resp = api.get_market_quote_ohlc(symbol=symbol_str, interval="1d", api_version="2.0")
    # SDK returns an object; .data is a dict-like mapping key → quote
    if resp and hasattr(resp, "data") and resp.data:
        return dict(resp.data)
    return {}


async def _fetch_ohlc_batch_async(token: str, keys: list[str]) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _fetch_ohlc_batch, token, keys)


@router.get("/market/movers")
async def market_movers(limit: int = Query(25, ge=1, le=100)):
    global _MOVERS_CACHE, _MOVERS_CACHE_AT

    # Serve from cache if fresh
    now = time.time()
    if _MOVERS_CACHE is not None and (now - _MOVERS_CACHE_AT) < _CACHE_TTL_SECONDS:
        return _MOVERS_CACHE

    token = await get_valid_token()
    if not token:
        return JSONResponse(status_code=401, content={"error": "not authenticated"})

    # Load universe (cached after first call)
    try:
        universe = await asyncio.get_running_loop().run_in_executor(None, _get_universe)
    except Exception as exc:
        logger.error(f"market/movers: universe download failed: {exc}")
        return JSONResponse(status_code=503, content={"error": f"universe unavailable: {exc}"})

    if not universe:
        return JSONResponse(status_code=503, content={"error": "empty universe"})

    # Query format is pipe: "NSE_EQ|INE002A01018". Response keys are colon: "NSE_EQ:RELIANCE".
    # Each response item has instrument_token in pipe format — use that to map back to metadata.
    pipe_keys: list[str] = []
    pipe_meta: dict[str, dict] = {}  # pipe key → {trading_symbol, name}
    seen: set[str] = set()
    for row in universe:
        sym = row.get("trading_symbol") or row.get("tradingsymbol") or ""
        name = row.get("name") or row.get("company_name") or sym
        ik = row.get("instrument_key") or ""
        if not ik or not ik.startswith("NSE_EQ|") or ik in seen:
            continue
        seen.add(ik)
        pipe_keys.append(ik)
        pipe_meta[ik] = {"trading_symbol": sym, "name": name}

    # Batch into groups of _BATCH_SIZE and fetch concurrently
    batches = [pipe_keys[i:i + _BATCH_SIZE]
               for i in range(0, len(pipe_keys), _BATCH_SIZE)]

    try:
        results = await asyncio.gather(
            *[_fetch_ohlc_batch_async(token, batch) for batch in batches],
            return_exceptions=True,
        )
    except Exception as exc:
        logger.error(f"market/movers: gather failed: {exc}")
        return JSONResponse(status_code=503, content={"error": str(exc)})

    # Merge all batch results
    merged: dict = {}
    for r in results:
        if isinstance(r, Exception):
            logger.warning(f"market/movers: a batch failed: {r}")
            continue
        merged.update(r)

    if not merged:
        return JSONResponse(status_code=503, content={"error": "no quotes returned"})

    # Compute change_pct for each stock.
    # Response keys are "NSE_EQ:SYMBOL" (colon); each quote has instrument_token in pipe format.
    stocks = []
    for _resp_key, quote in merged.items():
        try:
            ohlc = quote.ohlc if hasattr(quote, "ohlc") else quote.get("ohlc")
            ltp  = quote.last_price if hasattr(quote, "last_price") else quote.get("last_price")
            # instrument_token is the pipe-format key: "NSE_EQ|INE..."
            ik   = (quote.instrument_token if hasattr(quote, "instrument_token")
                    else quote.get("instrument_token", ""))

            if ohlc is None or ltp is None:
                continue

            open_px = ohlc.open if hasattr(ohlc, "open") else ohlc.get("open")
            high_px = ohlc.high if hasattr(ohlc, "high") else ohlc.get("high")
            low_px  = ohlc.low  if hasattr(ohlc, "low")  else ohlc.get("low")

            if open_px is None or open_px == 0:
                continue

            if ltp < 20:  # skip penny / illiquid stocks
                continue

            change_pct = (ltp - open_px) / open_px * 100

            meta = pipe_meta.get(ik, {})
            trading_symbol = meta.get("trading_symbol") or _resp_key.split(":", 1)[-1]
            name = meta.get("name", trading_symbol)

            stocks.append({
                "instrument_key": ik or _resp_key,
                "trading_symbol": trading_symbol,
                "name": name,
                "ltp": ltp,
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "change_pct": round(change_pct, 4),
            })
        except Exception as exc:
            logger.debug(f"market/movers: skipping {_resp_key}: {exc}")
            continue

    stocks.sort(key=lambda x: x["change_pct"], reverse=True)
    gainers = [s for s in stocks if s["change_pct"] > 0][:limit]
    losers  = [s for s in reversed(stocks) if s["change_pct"] < 0][:limit]

    payload = {
        "gainers": gainers,
        "losers": losers,
        "total_stocks": len(stocks),
        "computed_at": int(now),
    }

    _MOVERS_CACHE = payload
    _MOVERS_CACHE_AT = now

    return payload


# ── Stock search endpoint ──────────────────────────────────────────────────────

_STOP_WORDS = {
    "the","a","an","and","or","of","in","at","to","is","are","was","were","be",
    "been","has","have","had","that","this","for","on","with","as","by","from",
    "its","it","new","will","says","said","after","but","up","down","over",
    "ltd","limited","pvt","private","india","indian","co","corp","industries",
    "reports","quarterly","results","profit","loss","revenue","q1","q2","q3","q4",
}


@router.get("/market/search")
async def market_search(q: str = Query(..., min_length=2), limit: int = Query(6, ge=1, le=20)):
    """Search NSE universe by symbol or company name. Used for news → related scripts."""
    loop = asyncio.get_running_loop()
    universe = await loop.run_in_executor(None, _get_universe)

    q_lower = q.strip().lower()
    terms = [t for t in q_lower.split() if len(t) >= 3 and t not in _STOP_WORDS]
    if not terms:
        return {"results": []}

    scored: list[tuple[int, dict]] = []
    seen: set[str] = set()
    for row in universe:
        ik  = row.get("instrument_key", "")
        sym = (row.get("trading_symbol") or row.get("tradingsymbol") or "").lower()
        name = (row.get("name") or row.get("company_name") or "").lower()
        if not ik or ik in seen:
            continue

        score = 0
        for t in terms:
            if sym == t:          score += 10   # exact symbol match
            elif sym.startswith(t): score += 5
            elif t in sym:        score += 3
            if t in name:         score += 2

        if score > 0:
            seen.add(ik)
            scored.append((score, {
                "instrument_key":  ik,
                "trading_symbol":  row.get("trading_symbol") or row.get("tradingsymbol") or "",
                "name":            row.get("name") or row.get("company_name") or "",
            }))

    scored.sort(key=lambda x: x[0], reverse=True)
    return {"results": [r for _, r in scored[:limit]], "query": q}
