"""Stock detail endpoints.

GET /stock/{instrument_key:path}/candles?range=1d   — OHLCV candles for Lightweight Charts
GET /stock/{instrument_key:path}/info               — live quote + OHLC metadata
GET /stock/{instrument_key:path}/news?limit=20      — news & sentiment for a symbol
GET /stock/{instrument_key:path}/fundamentals       — company profile, ownership, key stats
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiosqlite
import upstox_client
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from src.db.database import get_valid_token
from src.utils.config import DB_PATH
from src.utils.logger import logger

router = APIRouter()

# ── Simple in-memory candle cache ─────────────────────────────────────────────
# (instrument_key, range) → (payload_dict, fetched_at_float)
_candle_cache: dict = {}

# ── Fundamentals cache: symbol → (payload, fetched_at) — 1-hour TTL ──────────
_fund_cache: dict = {}


# ── Technical helpers ──────────────────────────────────────────────────────────

def _rsi(closes: list[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    recent = changes[-period:]
    avg_gain = sum(max(0.0, c) for c in recent) / period
    avg_loss = sum(max(0.0, -c) for c in recent) / period
    if avg_loss == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def _ema(values: list[float], span: int) -> list[float]:
    k = 2 / (span + 1)
    ema = [values[0]]
    for v in values[1:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema


def _vwap(bars: list[dict]) -> Optional[float]:
    total_vol = sum(b["volume"] for b in bars)
    if total_vol == 0:
        return None
    return sum(((b["high"] + b["low"] + b["close"]) / 3) * b["volume"] for b in bars) / total_vol

_CACHE_TTL: dict[str, int] = {
    "1h":  30,
    "1d":  30,
    "1w":  120,
    "1m":  300,
    "1y":  600,
}

VALID_RANGES = {"1h", "1d", "1w", "1m", "1y"}


def _parse_candle_ts(ts_str: str) -> int:
    """Convert Upstox ISO-8601 timestamp to epoch seconds (floored to minute)."""
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    return (int(dt.timestamp()) // 60) * 60


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _date_str(days_ago: int) -> str:
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def _parse_raw_candles(raw: list) -> list[dict]:
    candles = []
    for c in raw:
        if len(c) < 6:
            continue
        try:
            ts = _parse_candle_ts(c[0])
        except (ValueError, AttributeError):
            continue
        candles.append({
            "time":   ts,
            "open":   c[1],
            "high":   c[2],
            "low":    c[3],
            "close":  c[4],
            "volume": int(c[5] or 0),
        })
    candles.sort(key=lambda x: x["time"])
    return candles


async def _fetch_upstox_intraday(
    instrument_key: str,
    unit: str,
    interval: str,
    token: str,
) -> list[dict]:
    """Fetch today's intraday candles via get_intra_day_candle_data (no date params)."""
    config = upstox_client.Configuration()
    config.access_token = token

    def _call():
        api = upstox_client.HistoryV3Api(upstox_client.ApiClient(config))
        return api.get_intra_day_candle_data(
            instrument_key=instrument_key,
            unit=unit,
            interval=interval,
        )

    loop = asyncio.get_running_loop()
    try:
        resp = await loop.run_in_executor(None, _call)
    except Exception as exc:
        logger.warning(f"stock/candles intraday: Upstox API error for {instrument_key}: {exc}")
        return []

    raw = getattr(getattr(resp, "data", None), "candles", None) or []
    return _parse_raw_candles(raw)


async def _fetch_upstox_candles(
    instrument_key: str,
    unit: str,
    interval: str,
    from_date: str,
    to_date: str,
    token: str,
) -> list[dict]:
    """Fetch historical candles via get_historical_candle_data1."""
    config = upstox_client.Configuration()
    config.access_token = token

    def _call():
        api = upstox_client.HistoryV3Api(upstox_client.ApiClient(config))
        return api.get_historical_candle_data1(
            instrument_key=instrument_key,
            unit=unit,
            interval=interval,
            to_date=to_date,
            from_date=from_date,
        )

    loop = asyncio.get_running_loop()
    try:
        resp = await loop.run_in_executor(None, _call)
    except Exception as exc:
        logger.warning(f"stock/candles: Upstox API error for {instrument_key}: {exc}")
        return []

    raw = getattr(getattr(resp, "data", None), "candles", None) or []
    return _parse_raw_candles(raw)


async def _fetch_live_bars(instrument_key: str, limit: Optional[int] = None) -> list[dict]:
    """Read today's 1-minute bars from bars_live table."""
    abs_path = os.path.abspath(DB_PATH)
    # today midnight UTC as epoch — we just use a wide window (last 24h) to catch IST day
    since = int(time.time()) - 86400
    async with aiosqlite.connect(abs_path) as db:
        async with db.execute(
            """
            SELECT minute_ts, open, high, low, close, volume
            FROM bars_live
            WHERE instrument_key = ? AND minute_ts >= ?
            ORDER BY minute_ts ASC
            """,
            (instrument_key, since),
        ) as cur:
            rows = await cur.fetchall()

    if limit is not None:
        rows = rows[-limit:]

    return [
        {
            "time":   row[0],
            "open":   row[1],
            "high":   row[2],
            "low":    row[3],
            "close":  row[4],
            "volume": row[5] or 0,
        }
        for row in rows
    ]


@router.get("/stock/{instrument_key:path}/candles")
async def stock_candles(
    instrument_key: str,
    range: str = Query("1d"),
):
    if range not in VALID_RANGES:
        return JSONResponse(
            status_code=400,
            content={"error": f"range must be one of {sorted(VALID_RANGES)}"},
        )

    # Check cache
    cache_key = (instrument_key, range)
    cached = _candle_cache.get(cache_key)
    if cached:
        payload, fetched_at = cached
        if time.time() - fetched_at < _CACHE_TTL[range]:
            return payload

    token = await get_valid_token()
    if not token:
        return JSONResponse(status_code=401, content={"error": "not authenticated"})

    candles: list[dict] = []

    if range in ("1h", "1d"):
        # Try bars_live first
        limit = 60 if range == "1h" else None
        candles = await _fetch_live_bars(instrument_key, limit=limit)

        if not candles:
            # Fall back to Upstox intraday endpoint (no date params required)
            candles = await _fetch_upstox_intraday(
                instrument_key=instrument_key,
                unit="minutes",
                interval="1",
                token=token,
            )
            if range == "1h":
                candles = candles[-60:]

    elif range == "1w":
        candles = await _fetch_upstox_candles(
            instrument_key=instrument_key,
            unit="minutes",
            interval="30",
            from_date=_date_str(7),
            to_date=_today_str(),
            token=token,
        )

    elif range == "1m":
        candles = await _fetch_upstox_candles(
            instrument_key=instrument_key,
            unit="days",
            interval="1",
            from_date=_date_str(30),
            to_date=_today_str(),
            token=token,
        )

    elif range == "1y":
        candles = await _fetch_upstox_candles(
            instrument_key=instrument_key,
            unit="days",
            interval="1",
            from_date=_date_str(365),
            to_date=_today_str(),
            token=token,
        )

    payload = {"candles": candles, "range": range}
    _candle_cache[cache_key] = (payload, time.time())
    return payload


# ── Info endpoint ──────────────────────────────────────────────────────────────

def _fetch_ohlc_quote(token: str, instrument_key: str) -> dict:
    """Synchronous: fetch OHLC quote for a single symbol. Run in executor."""
    cfg = upstox_client.Configuration()
    cfg.access_token = token
    api = upstox_client.MarketQuoteApi(upstox_client.ApiClient(cfg))
    resp = api.get_market_quote_ohlc(
        symbol=instrument_key,
        interval="1d",
        api_version="2.0",
    )
    if resp and hasattr(resp, "data") and resp.data:
        return dict(resp.data)
    return {}


def _get_meta_from_universe(instrument_key: str) -> dict:
    """Try to pull trading_symbol and name from the cached universe in market.py."""
    try:
        from src.api.market import _FULL_UNIVERSE
        if _FULL_UNIVERSE:
            for row in _FULL_UNIVERSE:
                ik = row.get("instrument_key") or ""
                if ik == instrument_key:
                    sym = row.get("trading_symbol") or row.get("tradingsymbol") or ""
                    name = row.get("name") or row.get("company_name") or sym
                    return {"trading_symbol": sym, "name": name}
    except Exception:
        pass
    return {}


@router.get("/stock/{instrument_key:path}/info")
async def stock_info(instrument_key: str):
    token = await get_valid_token()
    if not token:
        return JSONResponse(status_code=401, content={"error": "not authenticated"})

    loop = asyncio.get_running_loop()
    try:
        data = await loop.run_in_executor(None, _fetch_ohlc_quote, token, instrument_key)
    except Exception as exc:
        logger.warning(f"stock/info: OHLC fetch failed for {instrument_key}: {exc}")
        return JSONResponse(status_code=503, content={"error": str(exc)})

    if not data:
        return JSONResponse(status_code=404, content={"error": "no quote data"})

    # The response keys are "NSE_EQ:SYMBOL" (colon) — grab the first entry
    quote = None
    _resp_key = ""
    for _resp_key, v in data.items():
        quote = v
        break

    if quote is None:
        return JSONResponse(status_code=404, content={"error": "empty quote"})

    ltp = quote.last_price if hasattr(quote, "last_price") else quote.get("last_price")
    ohlc = quote.ohlc if hasattr(quote, "ohlc") else quote.get("ohlc")

    open_px = high_px = low_px = prev_close = None
    if ohlc is not None:
        open_px    = ohlc.open  if hasattr(ohlc, "open")  else ohlc.get("open")
        high_px    = ohlc.high  if hasattr(ohlc, "high")  else ohlc.get("high")
        low_px     = ohlc.low   if hasattr(ohlc, "low")   else ohlc.get("low")
        prev_close = ohlc.close if hasattr(ohlc, "close") else ohlc.get("close")

    change_pct = None
    if ltp is not None and open_px and open_px != 0:
        change_pct = round((ltp - open_px) / open_px * 100, 4)

    meta = _get_meta_from_universe(instrument_key)
    # Fall back: response key is "NSE_EQ:RELIANCE" — extract symbol from that
    trading_symbol = meta.get("trading_symbol") or _resp_key.split(":", 1)[-1]
    name = meta.get("name") or trading_symbol

    return {
        "instrument_key": instrument_key,
        "trading_symbol": trading_symbol,
        "name":           name,
        "ltp":            ltp,
        "open":           open_px,
        "high":           high_px,
        "low":            low_px,
        "prev_close":     prev_close,
        "change_pct":     change_pct,
    }


# ── News endpoint ──────────────────────────────────────────────────────────────

@router.get("/stock/{instrument_key:path}/news")
async def stock_news(
    instrument_key: str,
    limit: int = Query(20, ge=1, le=100),
):
    abs_path = os.path.abspath(DB_PATH)
    try:
        async with aiosqlite.connect(abs_path) as db:
            async with db.execute(
                """
                SELECT id, published_at, source, headline, url,
                       instrument_key, sentiment_score, sentiment_confidence
                FROM news
                WHERE instrument_key = ? OR instrument_key IS NULL
                ORDER BY published_at DESC
                LIMIT ?
                """,
                (instrument_key, limit),
            ) as cur:
                rows = await cur.fetchall()
    except Exception as exc:
        logger.error(f"stock/news: DB error: {exc}")
        return JSONResponse(status_code=500, content={"error": str(exc)})

    news = [
        {
            "id":                   row[0],
            "published_at":         row[1],
            "source":               row[2],
            "headline":             row[3],
            "url":                  row[4],
            "instrument_key":       row[5],
            "sentiment_score":      row[6],
            "sentiment_confidence": row[7],
        }
        for row in rows
    ]
    return {"news": news, "total": len(news)}


# ── Signal endpoint ───────────────────────────────────────────────────────────

@router.get("/stock/{instrument_key:path}/signal")
async def stock_signal(instrument_key: str):
    """Composite 1-hour directional score: ML model + RSI + momentum + VWAP + sentiment."""
    factors: list[dict] = []
    raw_score = 0.0
    weight_used = 0.0

    # ── 1. ML model prediction (209 streamed symbols only) ─────────────────
    try:
        from src.paper import loop as paper_loop
        engine = paper_loop.get_engine()
        if engine and engine.last_signals:
            pred = engine.last_signals.get(instrument_key)
            if pred is not None:
                # Predicted return is typically ±0.5%; scale to ±3 contribution
                model_contrib = max(-3.0, min(3.0, float(pred) * 600))
                raw_score   += model_contrib * 0.45
                weight_used += 0.45
                factors.append({
                    "name":   "ML model",
                    "score":  round(model_contrib, 2),
                    "detail": f"{float(pred) * 100:+.3f}% predicted return",
                    "icon":   "🤖",
                })
    except Exception:
        pass

    # ── 2. Technical signals from bars_live ────────────────────────────────
    bars = await _fetch_live_bars(instrument_key)
    closes = [b["close"] for b in bars]

    if len(closes) >= 5:
        ltp = closes[-1]

        # RSI (14-period) — 0.20 weight
        rsi_val = _rsi(closes, 14)
        if rsi_val is not None:
            if rsi_val >= 70:
                rsi_contrib = -2.5
                rsi_detail  = f"RSI {rsi_val:.0f} — overbought"
            elif rsi_val <= 30:
                rsi_contrib = 2.5
                rsi_detail  = f"RSI {rsi_val:.0f} — oversold"
            else:
                rsi_contrib = (50.0 - rsi_val) / 20.0   # ±2.5 at extremes
                rsi_detail  = f"RSI {rsi_val:.0f}"
            raw_score   += rsi_contrib * 0.20
            weight_used += 0.20
            factors.append({
                "name": "RSI (14)",
                "score": round(rsi_contrib, 2),
                "detail": rsi_detail,
                "icon": "📊",
            })

        # Short-term momentum: 5 min, 15 min, 30 min — 0.20 weight total
        mom_contrib = 0.0
        mom_parts   = []
        for n, label in ((5, "5m"), (15, "15m"), (30, "30m")):
            if len(closes) >= n + 1:
                pct = (closes[-1] - closes[-(n + 1)]) / closes[-(n + 1)] * 100
                mom_parts.append(f"{label}: {pct:+.2f}%")
                mom_contrib += max(-1.5, min(1.5, pct * 15))
        if mom_parts:
            mom_contrib = max(-3.0, min(3.0, mom_contrib / len(mom_parts)))
            raw_score   += mom_contrib * 0.20
            weight_used += 0.20
            factors.append({
                "name": "Momentum",
                "score": round(mom_contrib, 2),
                "detail": "  ·  ".join(mom_parts),
                "icon": "⚡",
            })

        # VWAP position — 0.10 weight
        vwap_val = _vwap(bars)
        if vwap_val and vwap_val > 0:
            vwap_pct     = (ltp - vwap_val) / vwap_val * 100
            vwap_contrib = max(-2.0, min(2.0, -vwap_pct * 20))  # above VWAP = slightly bearish (stretched)
            raw_score   += vwap_contrib * 0.10
            weight_used += 0.10
            direction    = "above" if ltp > vwap_val else "below"
            factors.append({
                "name": "vs VWAP",
                "score": round(vwap_contrib, 2),
                "detail": f"₹{ltp:.2f} {direction} VWAP ₹{vwap_val:.2f} ({vwap_pct:+.2f}%)",
                "icon": "📍",
            })

        # EMA crossover (9 vs 21) — 0.05 weight
        if len(closes) >= 22:
            ema9  = _ema(closes, 9)
            ema21 = _ema(closes, 21)
            cross = ema9[-1] - ema21[-1]
            cross_prev = ema9[-2] - ema21[-2]
            if cross > 0 and cross_prev <= 0:
                ema_contrib = 2.5
                ema_detail  = "EMA 9 crossed above 21 — bullish"
            elif cross < 0 and cross_prev >= 0:
                ema_contrib = -2.5
                ema_detail  = "EMA 9 crossed below 21 — bearish"
            else:
                ema_contrib = max(-1.5, min(1.5, cross / closes[-1] * 1000))
                ema_detail  = f"EMA9 {'>' if cross > 0 else '<'} EMA21"
            raw_score   += ema_contrib * 0.05
            weight_used += 0.05
            factors.append({
                "name": "EMA 9/21",
                "score": round(ema_contrib, 2),
                "detail": ema_detail,
                "icon": "〰️",
            })

    # ── 3. Sentiment ──────────────────────────────────────────────────────────
    try:
        from src.paper import loop as paper_loop
        engine = paper_loop.get_engine()
        if engine and engine.last_sentiment_scores:
            sent = engine.last_sentiment_scores.get(instrument_key)
            if sent is not None:
                sent_contrib = max(-2.0, min(2.0, float(sent) * 4))
                raw_score   += sent_contrib * 0.05
                weight_used += 0.05
                factors.append({
                    "name": "Sentiment",
                    "score": round(sent_contrib, 2),
                    "detail": f"News score {sent:+.2f}",
                    "icon": "📰",
                })
    except Exception:
        pass

    # ── Normalise to ±5 ───────────────────────────────────────────────────
    if weight_used > 0:
        score = max(-5.0, min(5.0, raw_score / weight_used * weight_used))
    else:
        score = 0.0

    if score >= 1.5:
        direction = "BULLISH"
        action    = "Consider Long ▲"
        color     = "green"
    elif score <= -1.5:
        direction = "BEARISH"
        action    = "Consider Short ▼"
        color     = "red"
    elif score >= 0.5:
        direction = "MILDLY BULLISH"
        action    = "Slight long bias"
        color     = "green"
    elif score <= -0.5:
        direction = "MILDLY BEARISH"
        action    = "Slight short bias"
        color     = "red"
    else:
        direction = "NEUTRAL"
        action    = "No clear signal — wait"
        color     = "muted"

    return {
        "score":          round(score, 2),
        "direction":      direction,
        "action":         action,
        "color":          color,
        "factors":        factors,
        "bars_count":     len(bars),
        "has_model":      any(f["name"] == "ML model" for f in factors),
        "computed_at":    int(time.time()),
    }


# ── Fundamentals endpoint ──────────────────────────────────────────────────────

def _fetch_fundamentals_sync(trading_symbol: str) -> dict:
    """Fetch company profile + key stats from Yahoo Finance (.NS ticker). Sync."""
    import yfinance as yf
    ticker = yf.Ticker(f"{trading_symbol}.NS")
    info = ticker.info

    # Market cap in INR → convert to crore
    market_cap_raw = info.get("marketCap")
    market_cap_cr  = round(market_cap_raw / 1e7, 1) if market_cap_raw else None

    # Promoter holding is approximated by insiders % for Indian stocks
    promoter_pct = info.get("heldPercentInsiders")
    if promoter_pct is not None:
        promoter_pct = round(promoter_pct * 100, 2)
    inst_pct = info.get("heldPercentInstitutions")
    if inst_pct is not None:
        inst_pct = round(inst_pct * 100, 2)
    public_pct = None
    if promoter_pct is not None and inst_pct is not None:
        public_pct = round(max(0.0, 100 - promoter_pct - inst_pct), 2)

    # Officers: name + title, skip duplicates
    officers = []
    seen_names: set = set()
    for o in (info.get("companyOfficers") or []):
        name = o.get("name", "").strip()
        title = o.get("title", "").strip()
        if name and name not in seen_names:
            seen_names.add(name)
            officers.append({"name": name, "title": title})

    # Business summary — truncate gracefully at sentence boundary
    summary = info.get("longBusinessSummary") or ""
    if len(summary) > 400:
        cut = summary.rfind(".", 0, 400)
        summary = summary[: cut + 1] if cut > 100 else summary[:400] + "…"

    return {
        "trading_symbol":    trading_symbol,
        "sector":            info.get("sectorDisp") or info.get("sector"),
        "industry":          info.get("industryDisp") or info.get("industry"),
        "website":           info.get("website"),
        "summary":           summary,
        "full_summary":      info.get("longBusinessSummary") or "",
        "market_cap_cr":     market_cap_cr,
        "pe_ratio":          round(info.get("trailingPE"), 2) if info.get("trailingPE") else None,
        "week52_low":        info.get("fiftyTwoWeekLow"),
        "week52_high":       info.get("fiftyTwoWeekHigh"),
        "employees":         info.get("fullTimeEmployees"),
        "promoter_pct":      promoter_pct,
        "institutional_pct": inst_pct,
        "public_pct":        public_pct,
        "officers":          officers[:6],
    }


@router.get("/stock/{instrument_key:path}/fundamentals")
async def stock_fundamentals(instrument_key: str):
    # Resolve trading_symbol — try universe first, then force-load it
    meta = _get_meta_from_universe(instrument_key)
    trading_symbol = meta.get("trading_symbol", "")
    if not trading_symbol or trading_symbol.startswith("INE"):
        # Universe not loaded yet — trigger load now
        loop = asyncio.get_running_loop()
        try:
            from src.api.market import _get_universe
            universe = await loop.run_in_executor(None, _get_universe)
            for row in universe:
                if row.get("instrument_key") == instrument_key:
                    trading_symbol = row.get("trading_symbol") or row.get("tradingsymbol") or ""
                    break
        except Exception:
            pass
    if not trading_symbol or len(trading_symbol) > 20 or trading_symbol.startswith("INE"):
        return JSONResponse(status_code=400, content={"error": "could not resolve trading symbol"})

    cache_key = trading_symbol.upper()
    cached = _fund_cache.get(cache_key)
    if cached:
        payload, fetched_at = cached
        if time.time() - fetched_at < 3600:
            return payload

    loop = asyncio.get_running_loop()
    try:
        payload = await loop.run_in_executor(None, _fetch_fundamentals_sync, trading_symbol)
    except Exception as exc:
        logger.warning(f"fundamentals: yfinance failed for {trading_symbol}: {exc}")
        return JSONResponse(status_code=503, content={"error": str(exc)})

    _fund_cache[cache_key] = (payload, time.time())
    return payload
