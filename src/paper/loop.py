"""Paper-trading engine loop.

One async task that runs every PAPER_LOOP_INTERVAL_SECONDS and does, in order:

    1. Mark-to-market every open position against the latest LTP.
       Close on SL/TP/EOD triggers.
    2. Check kill switches (daily P&L cap). On trip, set halt = True.
    3. At minute boundary (only when not halted), call decide() to generate
       new intents. Queue them.
    4. Fill any pending intents at the current LTP.

This is deliberately one task, not two — the cadence is fine grained enough
(every 2s) that both mark-to-market and decisions are responsive without
needing concurrent loops.
"""

from __future__ import annotations

import asyncio
import time
from typing import List, Optional

from src.api import streamer_manager
from src.bot.engine import (
    EngineConfig,
    OrderIntent,
    decide,
    is_forced_exit,
)
from src.features.technical import IST_OFFSET_SECONDS
from src.model.infer import ModelArtifact, load_model, score as _model_score
from src.model.train import paths_for_name
from src.paper.executor import PaperExecutor
from src.features.sentiment_cache import live_sentiment_by_symbol
from src.paper.feature_cache import build_live_feature_frame
from src.paper.persistence import (
    get_halt_state,
    list_closed_positions_today,
    list_open_positions,
    set_halt,
    todays_realised_pnl,
    update_stop_loss_price,
)
import aiosqlite
from src.bot.costs import round_trip_cost_at_price
from src.utils.config import DB_PATH
from src.utils.logger import logger

NIFTY_KEY = "NSE_INDEX|Nifty 50"
REGIME_DOWN_PCT = 0.001   # >0.1% below open → suppress longs
REGIME_UP_PCT   = 0.0015  # >0.15% above open → suppress shorts


# Lazily-loaded {instrument_key: trading_symbol} so the dashboard can show
# 'RELIANCE' instead of 'NSE_EQ|INE002A01018'. Populated on first call.
_SYMBOL_LOOKUP: Optional[dict] = None


def _symbol_lookup() -> dict:
    global _SYMBOL_LOOKUP
    if _SYMBOL_LOOKUP is None:
        try:
            from src.data.universe import load_universe
            _SYMBOL_LOOKUP = {u["instrument_key"]: u["trading_symbol"] for u in load_universe()}
        except Exception:
            _SYMBOL_LOOKUP = {}
    return _SYMBOL_LOOKUP


async def _fetch_nifty_session_open(now_ts: int) -> Optional[float]:
    """First bar close for Nifty 50 today, used as the session-open reference."""
    from src.features.technical import IST_OFFSET_SECONDS, SECONDS_PER_DAY
    ist_today = (now_ts + IST_OFFSET_SECONDS) // SECONDS_PER_DAY
    session_open_ts = ist_today * SECONDS_PER_DAY - IST_OFFSET_SECONDS + 9 * 3600 + 15 * 60
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT close FROM bars_live WHERE instrument_key = ? AND minute_ts >= ? ORDER BY minute_ts ASC LIMIT 1",
            (NIFTY_KEY, session_open_ts),
        ) as cur:
            row = await cur.fetchone()
    return float(row[0]) if row else None


def _seconds_to_market_close(now_ts: float) -> Optional[int]:
    """Seconds until 15:30 IST today, or None if market is already closed."""
    ist = time.gmtime(now_ts + IST_OFFSET_SECONDS)
    if ist.tm_wday >= 5:
        return None
    minutes = ist.tm_hour * 60 + ist.tm_min
    close_minute = 15 * 60 + 30
    if minutes >= close_minute:
        return None
    return (close_minute - minutes) * 60 - ist.tm_sec

def _load_paper_symbols() -> list:
    """Full F&O equity universe from universe.json. Falls back to 5 large-caps."""
    try:
        from src.data.universe import load_universe
        return [u["instrument_key"] for u in load_universe()]
    except Exception:
        return [
            "NSE_EQ|INE002A01018",
            "NSE_EQ|INE040A01034",
            "NSE_EQ|INE009A01021",
            "NSE_EQ|INE467B01029",
            "NSE_EQ|INE090A01021",
        ]

PAPER_SYMBOLS: list = _load_paper_symbols()

PAPER_LOOP_INTERVAL_SECONDS = 2
DAILY_LOSS_CAP_INR = 999_999.0   # no effective cap
PAPER_MODEL_NAME = "v1"           # which model artefact to load

# Capital: ₹25,000 real equity × 5× MIS leverage = ₹1,25,000 buying power.
# Split evenly across 10 concurrent positions → ₹12,500 notional cap each.
# Daily loss cap = ₹1,000 (4% of equity). Stop trading if breached.
PAPER_ENGINE_CONFIG = EngineConfig(
    capital_inr=12_500.0,                        # ₹12,500 per position (₹25k equity × 5× MIS ÷ 10 slots)
    max_concurrent_positions=6,
    top_k_long=3,
    top_k_short=3,
    min_predicted_edge=0.0005,                   # raised from 0.00005 — require real edge, not noise
    stop_loss_pct=0.01,                          # 1% stop (was 0.5% — too tight for open volatility)
    target_pct=0.025,
    entry_window_open_minute_ist=9 * 60 + 45,    # no entries in first 30 min (was 09:30)
    entry_window_close_minute_ist=13 * 60 + 30,  # last entry 13:30 (was 14:30 — avoids pre-close chaos)
    invert_signals=False,                        # follow model signals directly (was True — hurt on up-days)
)


class PaperEngine:
    """Stateful per-process paper engine. Owned by FastAPI lifespan."""

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        model_name: str = PAPER_MODEL_NAME,
        interval_seconds: float = PAPER_LOOP_INTERVAL_SECONDS,
        daily_loss_cap_inr: float = DAILY_LOSS_CAP_INR,
    ):
        self.config = config or EngineConfig()
        self.model_name = model_name
        self.interval_seconds = interval_seconds
        self.daily_loss_cap_inr = daily_loss_cap_inr
        self.model: Optional[ModelArtifact] = None
        self.executor: Optional[PaperExecutor] = None
        self._task: Optional[asyncio.Task] = None
        self._pending_intents: List[OrderIntent] = []
        self._pending_sentiment: dict = {}
        self._last_decision_minute: int = -1
        self._session_started_for_date: Optional[int] = None
        # Runtime stats surfaced via /paper/status:
        self.cycles_run: int = 0
        self.last_cycle_at: float = 0.0
        self.last_skip_reasons: dict = {}
        # Cached per-symbol predictions from the last decision cycle (for /paper/signals).
        self.last_signals: dict = {}
        self.last_sentiment_scores: dict = {}
        self.last_decision_ts: float = 0.0
        # Nifty 50 session-open price for market-regime filter.
        self._nifty_session_open: Optional[float] = None
        self._nifty_session_date: int = -1
        # Extra headroom added by the user via /paper/extend-loss-cap.
        self.loss_cap_extension: float = 0.0
        # Timestamp of last REST quote refresh (for positions missing stream quotes).
        self._last_rest_quote_ts: float = 0.0
        # High-water mark per open position: (instrument_key, entry_ts) → peak INR profit.
        # Used by the profit-trail exit (50% pullback from peak, min ₹50 floor to arm).
        self._peak_profit_inr: dict = {}

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        try:
            model_path, metrics_path = paths_for_name(self.model_name)
            self.model = load_model(model_path=model_path, metrics_path=metrics_path)
        except FileNotFoundError as e:
            logger.warning(f"Paper engine not started: {e}")
            return
        self.executor = PaperExecutor(config=self.config, model_name=self.model_name)
        self._task = asyncio.create_task(self._run(), name="paper_engine")
        logger.info(
            f"Paper engine started (model={self.model_name}, "
            f"symbols={len(PAPER_SYMBOLS)}, loss_cap=₹{self.daily_loss_cap_inr:.0f})"
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):
            pass
        self._task = None
        logger.info("Paper engine stopped")

    async def status(self) -> dict:
        """Snapshot for the /paper/status endpoint."""
        now = time.time()
        open_positions = await list_open_positions()
        closed_today = await list_closed_positions_today(now)
        pnl_today = await todays_realised_pnl(now)
        halt = await get_halt_state(now)
        symbols = _symbol_lookup()
        unrealised = 0.0
        for p in open_positions:
            quote = streamer_manager.last_quote_by_symbol.get(p.instrument_key)
            if quote is not None:
                unrealised += p.unrealised_pnl_inr(quote)

        # Cumulative P&L series, one point per closed trade in entry order.
        # The dashboard renders this as a sparkline.
        pnl_series = []
        running = 0.0
        for p in sorted(closed_today, key=lambda x: x.exit_ts or 0):
            running += (p.realised_pnl_inr or 0.0)
            pnl_series.append({
                "exit_ts": p.exit_ts,
                "cum_pnl": round(running, 2),
            })

        wins = sum(1 for p in closed_today if (p.realised_pnl_inr or 0) > 0)
        losses = sum(1 for p in closed_today if (p.realised_pnl_inr or 0) < 0)

        return {
            "running": self._task is not None and not self._task.done(),
            "halted": halt["halted"],
            "halt_reason": halt["halt_reason"],
            "model": self.model_name,
            "open_positions": [
                {
                    "instrument_key": p.instrument_key,
                    "trading_symbol": symbols.get(p.instrument_key, p.instrument_key),
                    "side": p.side,
                    "qty": p.qty,
                    "entry_ts": p.entry_ts,
                    "entry_price": p.entry_price,
                    "stop_loss_price": p.stop_loss_price,
                    "target_price": p.target_price,
                    "last_quote": streamer_manager.last_quote_by_symbol.get(p.instrument_key),
                    "unrealised_pnl_inr": (
                        p.unrealised_pnl_inr(streamer_manager.last_quote_by_symbol[p.instrument_key])
                        if p.instrument_key in streamer_manager.last_quote_by_symbol
                        else None
                    ),
                    "estimated_cost_inr": round(
                        round_trip_cost_at_price(p.qty, p.entry_price, p.entry_price, p.side).total,
                        2,
                    ),
                    "predicted_return": p.predicted_return,
                    "model_name": p.model_name,
                    "entry_sentiment_score": p.entry_sentiment_score,
                }
                for p in open_positions
            ],
            "closed_trades_today": [
                {
                    "instrument_key": p.instrument_key,
                    "trading_symbol": symbols.get(p.instrument_key, p.instrument_key),
                    "side": p.side,
                    "qty": p.qty,
                    "entry_ts": p.entry_ts,
                    "exit_ts": p.exit_ts,
                    "entry_price": p.entry_price,
                    "exit_price": p.exit_price,
                    "exit_reason": p.exit_reason,
                    "realised_pnl_inr": p.realised_pnl_inr,
                    "actual_cost_inr": (
                        round(
                            p.qty * (
                                (p.exit_price - p.entry_price) if p.side == "long"
                                else (p.entry_price - p.exit_price)
                            ) - (p.realised_pnl_inr or 0),
                            2,
                        )
                        if p.exit_price is not None and p.realised_pnl_inr is not None
                        else None
                    ),
                    "predicted_return": p.predicted_return,
                    "model_name": p.model_name,
                    "entry_sentiment_score": p.entry_sentiment_score,
                }
                for p in sorted(closed_today, key=lambda x: x.exit_ts or 0, reverse=True)
            ],
            "pnl_series": pnl_series,
            "trades_today": len(closed_today),
            "wins_today": wins,
            "losses_today": losses,
            "realised_pnl_inr": pnl_today,
            "unrealised_pnl_inr": unrealised,
            "net_pnl_inr": pnl_today + unrealised,
            "cycles_run": self.cycles_run,
            "last_cycle_at": self.last_cycle_at,
            "last_skip_reasons": self.last_skip_reasons,
            "last_decision_ts": self.last_decision_ts,
            "top_signals": sorted(
                [{"k": k, "p": round(v, 5)} for k, v in self.last_signals.items()],
                key=lambda x: abs(x["p"]),
                reverse=True,
            )[:8],
            "now_ts": now,
            "seconds_to_market_close": _seconds_to_market_close(now),
            "market_open": streamer_manager.is_market_open(now),
            "daily_loss_cap_inr": self.daily_loss_cap_inr,
            "loss_cap_extension_inr": self.loss_cap_extension,
            "min_predicted_edge": self.config.min_predicted_edge,
        }

    # ------------------------------------------------------------------
    # Loop internals
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.interval_seconds)
                try:
                    await self._cycle_once()
                except Exception as e:
                    logger.error(f"Paper engine cycle error: {type(e).__name__}: {e}")
        except asyncio.CancelledError:
            return

    async def _refresh_rest_quotes(self, positions: list) -> None:
        """Batch-fetch LTPs via Upstox REST for positions with no stream quote yet."""
        from src.db.database import get_valid_token
        import upstox_client
        token = await get_valid_token()
        if not token:
            return
        keys = [p.instrument_key for p in positions]

        def _fetch():
            cfg = upstox_client.Configuration()
            cfg.access_token = token
            api = upstox_client.MarketQuoteApi(upstox_client.ApiClient(cfg))
            resp = api.get_market_quote_ohlc(
                symbol=",".join(keys), interval="1d", api_version="2.0"
            )
            if not (resp and hasattr(resp, "data") and resp.data):
                return
            for q in resp.data.values():
                ik = q.instrument_token if hasattr(q, "instrument_token") else None
                lp = q.last_price if hasattr(q, "last_price") else None
                if ik and lp:
                    streamer_manager.last_quote_by_symbol[ik] = float(lp)

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _fetch)
            filled = sum(1 for p in positions
                        if p.instrument_key in streamer_manager.last_quote_by_symbol)
            if filled:
                logger.info(f"REST quote refresh: filled {filled}/{len(positions)} missing quotes")
        except Exception as e:
            logger.warning(f"REST quote refresh failed: {type(e).__name__}: {e}")

    async def _cycle_once(self) -> None:
        now = time.time()
        now_int = int(now)
        self.cycles_run += 1
        self.last_cycle_at = now

        if not streamer_manager.is_market_open(now):
            return

        # Re-hydrate open positions from DB each cycle. Cheap, and means a
        # crashed-then-restarted bot picks up where it left off.
        open_positions = await list_open_positions()

        # Refresh quotes via REST for positions the stream hasn't ticked yet.
        # Runs at most once every 30s to avoid hammering the Upstox API.
        missing = [p for p in open_positions
                   if p.instrument_key not in streamer_manager.last_quote_by_symbol]
        if missing and now - self._last_rest_quote_ts > 30:
            self._last_rest_quote_ts = now
            await self._refresh_rest_quotes(missing)

        # 1. Mark-to-market: close any position whose live LTP has crossed SL/TP,
        #    or close everything at forced-exit minute (14:55 IST).
        forced = is_forced_exit(now_int, self.config)
        for pos in open_positions:
            quote = streamer_manager.last_quote_by_symbol.get(pos.instrument_key)
            if quote is None:
                continue
            reason: Optional[str] = None
            if forced:
                reason = "eod"
            elif pos.side == "long":
                if quote <= pos.stop_loss_price:
                    reason = "stop_loss"
                elif quote >= pos.target_price:
                    reason = "target"
            else:  # short
                if quote >= pos.stop_loss_price:
                    reason = "stop_loss"
                elif quote <= pos.target_price:
                    reason = "target"
            # Time stop: exit if open > 15 min and price hasn't moved > 0.1%.
            if reason is None:
                age_minutes = (now_int - pos.entry_ts) / 60
                pnl_pct = abs(quote - pos.entry_price) / pos.entry_price
                if age_minutes >= 15 and pnl_pct < 0.001:
                    reason = "time_stop"

            # Trailing stop: once at ≥50% of target gain, move stop to break-even.
            # Once at ≥75% of target, lock in 25% of target gain.
            if reason is None:
                half_tgt = self.config.target_pct / 2
                three_q_tgt = self.config.target_pct * 0.75
                quarter_tgt = self.config.target_pct * 0.25
                if pos.side == "long":
                    gain_pct = (quote - pos.entry_price) / pos.entry_price
                    if gain_pct >= three_q_tgt:
                        lock_sl = pos.entry_price * (1 + quarter_tgt)
                        if pos.stop_loss_price < lock_sl:
                            await update_stop_loss_price(pos.instrument_key, pos.entry_ts, lock_sl)
                            logger.info(f"Trailing stop ▲ {_symbol_lookup().get(pos.instrument_key, pos.instrument_key)}: stop locked at +{quarter_tgt*100:.2f}% ({lock_sl:.2f})")
                    elif gain_pct >= half_tgt and pos.stop_loss_price < pos.entry_price:
                        await update_stop_loss_price(pos.instrument_key, pos.entry_ts, pos.entry_price)
                        logger.info(f"Trailing stop ▲ {_symbol_lookup().get(pos.instrument_key, pos.instrument_key)}: stop moved to break-even ({pos.entry_price:.2f})")
                else:  # short
                    gain_pct = (pos.entry_price - quote) / pos.entry_price
                    if gain_pct >= three_q_tgt:
                        lock_sl = pos.entry_price * (1 - quarter_tgt)
                        if pos.stop_loss_price > lock_sl:
                            await update_stop_loss_price(pos.instrument_key, pos.entry_ts, lock_sl)
                            logger.info(f"Trailing stop ▼ {_symbol_lookup().get(pos.instrument_key, pos.instrument_key)}: stop locked at +{quarter_tgt*100:.2f}% ({lock_sl:.2f})")
                    elif gain_pct >= half_tgt and pos.stop_loss_price > pos.entry_price:
                        await update_stop_loss_price(pos.instrument_key, pos.entry_ts, pos.entry_price)
                        logger.info(f"Trailing stop ▼ {_symbol_lookup().get(pos.instrument_key, pos.instrument_key)}: stop moved to break-even ({pos.entry_price:.2f})")

            # Profit-trail exit: if the trade has ever been up ≥ ₹50 and the current
            # unrealised profit has pulled back to ≤ 50% of that peak, exit now.
            # Arms only above the ₹50 floor so micro-fluctuations don't trigger it.
            if reason is None:
                pk = (pos.instrument_key, pos.entry_ts)
                cur_pnl = pos.unrealised_pnl_inr(quote)
                prev_peak = self._peak_profit_inr.get(pk, 0.0)
                if cur_pnl > prev_peak:
                    self._peak_profit_inr[pk] = cur_pnl
                    prev_peak = cur_pnl
                # Only trail once the position is meaningfully in profit (2.4% of ₹12,500 notional).
                # Then exit only on a 25% pullback from the peak, giving winners room to run.
                PROFIT_TRAIL_FLOOR_INR = 300.0
                if prev_peak >= PROFIT_TRAIL_FLOOR_INR and cur_pnl <= prev_peak * 0.75:
                    sym = _symbol_lookup().get(pos.instrument_key, pos.instrument_key)
                    logger.info(
                        f"Profit trail exit {sym}: peak ₹{prev_peak:.0f} → now ₹{cur_pnl:.0f} "
                        f"(12.5% pullback from high)"
                    )
                    reason = "profit_trail"

            if reason is not None:
                self._peak_profit_inr.pop((pos.instrument_key, pos.entry_ts), None)
                await self.executor.close_position(pos, exit_ts=now_int, exit_price=quote, reason=reason)

        # 2. Kill switch: daily loss cap. Once tripped, halt for the day.
        halt = await get_halt_state(now)
        if not halt["halted"]:
            pnl_today = await todays_realised_pnl(now)
            effective_cap = self.daily_loss_cap_inr + self.loss_cap_extension
            if pnl_today <= -effective_cap:
                await set_halt(
                    reason=f"daily_loss_cap_breached ({pnl_today:.2f} <= -{effective_cap:.0f})",
                    halted=True,
                    now_ts=now,
                )
                logger.warning(
                    f"Daily loss cap breached at ₹{pnl_today:.2f}; halting new entries for today."
                )
                halt = await get_halt_state(now)

        # Forced exit window: no new entries regardless of halt state.
        if forced:
            return

        # 3. Decision happens on minute boundary only, and only when not halted.
        minute_now = now_int // 60
        if minute_now == self._last_decision_minute:
            # Fill pending intents from the previous minute's decision.
            await self._fill_pending_intents(now_int)
            return
        self._last_decision_minute = minute_now

        if halt["halted"]:
            return

        # Build feature frame for the full universe. Run in a thread so the
        # 5-10s compute doesn't block the event loop (SL/TP monitoring keeps running).
        loop = asyncio.get_event_loop()
        features = await loop.run_in_executor(
            None, build_live_feature_frame, PAPER_SYMBOLS
        )
        if features.empty:
            self.last_skip_reasons = {"no_features_yet": 1}
            return

        # Sentiment scores: sync sqlite read, fast enough to run inline.
        sentiment = live_sentiment_by_symbol(PAPER_SYMBOLS, now_ts=now_int)
        if sentiment:
            logger.debug(f"Sentiment scores: {sentiment}")

        # Cache all-symbol model predictions so /paper/signals can serve without re-running inference.
        try:
            preds = _model_score(self.model, features)
            self.last_signals = {
                str(ik): float(p)
                for ik, p in zip(features["instrument_key"], preds)
            }
            self.last_sentiment_scores = dict(sentiment)
            self.last_decision_ts = now
        except Exception as e:
            logger.error(f"Signal cache failed: {type(e).__name__}: {e}")

        # Re-read state so closed positions from this cycle are visible.
        open_positions = await list_open_positions()
        closed_today = await list_closed_positions_today(now)

        allow_longs, allow_shorts = await self._market_regime(now_int)

        result = decide(
            features_at_minute=features,
            model=self.model,
            open_positions=open_positions,
            config=self.config,
            now_ts=now_int,
            closed_positions=closed_today,
            sentiment_scores=sentiment,
            allow_longs=allow_longs,
            allow_shorts=allow_shorts,
        )
        self.last_skip_reasons = dict(result.skipped_reasons)

        # Queue entry intents to fill next cycle (matches backtest's
        # decision-at-T -> fill-at-T+1 convention).
        for intent in result.intents:
            if intent.reason in ("exit_eod", "exit_kill_switch"):
                continue  # eod is handled by the forced-exit branch above
            self._pending_intents.append(intent)
            self._pending_sentiment[intent.instrument_key] = sentiment.get(intent.instrument_key)
        if result.intents:
            logger.info(
                f"Decision @ minute {minute_now}: {len(result.intents)} intent(s), "
                f"skips={result.skipped_reasons}"
            )

    async def _market_regime(self, now_int: int) -> tuple:
        """Return (allow_longs, allow_shorts) based on Nifty 50 vs session open."""
        nifty_ltp = streamer_manager.last_quote_by_symbol.get(NIFTY_KEY)
        if nifty_ltp is None:
            return True, True

        from src.features.technical import IST_OFFSET_SECONDS, SECONDS_PER_DAY
        today = (now_int + IST_OFFSET_SECONDS) // SECONDS_PER_DAY
        if self._nifty_session_date != today or self._nifty_session_open is None:
            open_price = await _fetch_nifty_session_open(now_int)
            self._nifty_session_open = open_price if open_price is not None else nifty_ltp
            self._nifty_session_date = today

        ref = self._nifty_session_open
        if not ref or ref <= 0:
            return True, True

        nifty_return = (nifty_ltp - ref) / ref
        if nifty_return <= -REGIME_DOWN_PCT:
            logger.debug(f"Regime: Nifty {nifty_return:.2%} from open — longs suppressed")
            return False, True
        if nifty_return >= REGIME_UP_PCT:
            logger.debug(f"Regime: Nifty {nifty_return:.2%} from open — shorts suppressed")
            return True, False
        return True, True

    async def _fill_pending_intents(self, now_int: int) -> None:
        if not self._pending_intents:
            return
        unfilled: List[OrderIntent] = []
        # Re-check halt: if it tripped between decision and fill, drop intents.
        halt = await get_halt_state(now_int)
        if halt["halted"]:
            self._pending_intents.clear()
            return
        for intent in self._pending_intents:
            quote = streamer_manager.last_quote_by_symbol.get(intent.instrument_key)
            if quote is None:
                unfilled.append(intent)
                continue
            try:
                await self.executor.fill_intent(
                    intent,
                    fill_ts=now_int,
                    last_quote_price=quote,
                    entry_sentiment_score=self._pending_sentiment.get(intent.instrument_key),
                )
            except Exception as e:
                logger.error(f"Paper fill failed for {intent.instrument_key}: {type(e).__name__}: {e}")
        self._pending_intents = unfilled
        self._pending_sentiment = {i.instrument_key: self._pending_sentiment.get(i.instrument_key) for i in unfilled}


# Process-global singleton — owned by FastAPI lifespan, surfaced to HTTP handlers.
_engine: Optional[PaperEngine] = None


def get_engine() -> Optional[PaperEngine]:
    return _engine


async def start_paper_engine(**kwargs) -> None:
    global _engine
    if _engine is None:
        kwargs.setdefault("config", PAPER_ENGINE_CONFIG)
        _engine = PaperEngine(**kwargs)
    await _engine.start()


async def stop_paper_engine() -> None:
    global _engine
    if _engine is not None:
        await _engine.stop()
