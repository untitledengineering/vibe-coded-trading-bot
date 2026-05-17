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
)
from src.utils.logger import logger


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
DAILY_LOSS_CAP_INR = 1000.0
PAPER_MODEL_NAME = "v1"   # which model artefact to load

# With 209 symbols the model has real signal breadth. Run up to 10 concurrent
# positions (up from 4) and allow more picks per cycle.
PAPER_ENGINE_CONFIG = EngineConfig(
    max_concurrent_positions=10,
    top_k_long=4,
    top_k_short=4,
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
            "now_ts": now,
            "seconds_to_market_close": _seconds_to_market_close(now),
            "market_open": streamer_manager.is_market_open(now),
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
            if reason is not None:
                await self.executor.close_position(pos, exit_ts=now_int, exit_price=quote, reason=reason)

        # 2. Kill switch: daily loss cap. Once tripped, halt for the day.
        halt = await get_halt_state(now)
        if not halt["halted"]:
            pnl_today = await todays_realised_pnl(now)
            if pnl_today <= -self.daily_loss_cap_inr:
                await set_halt(
                    reason=f"daily_loss_cap_breached ({pnl_today:.2f} <= -{self.daily_loss_cap_inr:.0f})",
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

        result = decide(
            features_at_minute=features,
            model=self.model,
            open_positions=open_positions,
            config=self.config,
            now_ts=now_int,
            closed_positions=closed_today,
            sentiment_scores=sentiment,
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
