"""Decision engine.

This is the SINGLE source of truth for what trades the bot wants to make. The
backtester replays bars and asks this function; the live paper engine asks
this function. Their answers must be identical for the same inputs — that's
the whole point of having one function.

Inputs are all explicit (no globals): current features for every candidate
symbol, the open positions, the model, the config. Output is a list of
OrderIntent objects. The executor (backtest or paper) is responsible for
turning intents into fills.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src.bot.positions import Position, Side
from src.model.infer import ModelArtifact, score


@dataclass(frozen=True)
class EngineConfig:
    # Risk shape
    capital_inr: float = 25_000.0          # per-position notional cap; not the full equity
    max_concurrent_positions: int = 4
    stop_loss_pct: float = 0.005           # -0.5% from entry
    target_pct: float = 0.008              # +0.8% from entry

    # Selection
    min_predicted_edge: float = 0.0015     # |predicted return| must clear cost floor (~0.15%)
    top_k_long: int = 2                    # at most this many long intents per cycle
    top_k_short: int = 2                   # at most this many short intents per cycle

    # Anti-churn (added 2026-05-15 after model_v1 backtest showed the engine
    # trading IDFC First Bank 17 times in a single session, re-entering 1 min
    # after each stop-out). Both filters check `closed_positions` for the day.
    cooldown_minutes: int = 30             # min minutes between successive trades on the same symbol
    max_trades_per_symbol_per_day: int = 3 # belt-and-suspenders cap

    # Hard exit and entry windows (epoch-seconds-of-day, IST). 09:15-15:30 IST = 09:15..15:30.
    entry_window_open_minute_ist: int = 9 * 60 + 30      # delay 15 min from open to skip volatility
    entry_window_close_minute_ist: int = 14 * 60 + 30    # last entry 14:30 IST
    forced_exit_minute_ist: int = 14 * 60 + 55           # square-off 14:55 IST

    # Sentiment veto: block new entries when news strongly contradicts direction.
    # A symbol's weighted sentiment score must be below -threshold to veto longs,
    # or above +threshold to veto shorts. 0.0 disables the veto entirely.
    sentiment_veto_threshold: float = 0.3

    # Invert model signals: place short when model says long, and vice versa.
    invert_signals: bool = False


@dataclass(frozen=True)
class OrderIntent:
    instrument_key: str
    side: Side
    qty: int
    reason: str               # "long_top_pick" | "short_bottom_pick" | "exit_eod" | "exit_kill_switch"
    predicted_return: float   # model's E[return_15m]; useful for the report
    # Optional absolute price levels set by rule-based strategies. When None,
    # the executor falls back to config.stop_loss_pct + target_pct.
    stop_loss_price: Optional[float] = None
    target_price: Optional[float] = None


@dataclass
class EngineCycleResult:
    intents: List[OrderIntent] = field(default_factory=list)
    skipped_reasons: Dict[str, int] = field(default_factory=dict)

    def add(self, intent: OrderIntent) -> None:
        self.intents.append(intent)

    def skip(self, reason: str) -> None:
        self.skipped_reasons[reason] = self.skipped_reasons.get(reason, 0) + 1


def _ist_minute_of_day(ts: int) -> int:
    """Minute-of-day in IST. 09:15 IST -> 555."""
    from src.features.technical import IST_OFFSET_SECONDS, SECONDS_PER_DAY
    ist_seconds = (ts + IST_OFFSET_SECONDS) % SECONDS_PER_DAY
    return ist_seconds // 60


def is_entry_window(ts: int, config: EngineConfig) -> bool:
    """True if `ts` is inside the bot's entry window for new positions."""
    minute = _ist_minute_of_day(ts)
    return config.entry_window_open_minute_ist <= minute <= config.entry_window_close_minute_ist


def is_forced_exit(ts: int, config: EngineConfig) -> bool:
    """True if `ts` is at or past the hard square-off minute."""
    return _ist_minute_of_day(ts) >= config.forced_exit_minute_ist


def _qty_for_notional(notional: float, price: float) -> int:
    if price <= 0:
        return 0
    return max(0, int(notional // price))


def _ist_session_date(ts: int) -> int:
    """Integer 'IST days since epoch'. Two timestamps in the same NSE trading
    session share the same value."""
    from src.features.technical import IST_OFFSET_SECONDS, SECONDS_PER_DAY
    return (ts + IST_OFFSET_SECONDS) // SECONDS_PER_DAY


def _last_exit_ts_by_symbol(closed_positions: List[Position]) -> Dict[str, int]:
    """Most recent exit_ts per instrument_key. Used for the cooldown filter."""
    out: Dict[str, int] = {}
    for p in closed_positions:
        if p.exit_ts is None:
            continue
        prev = out.get(p.instrument_key, 0)
        if p.exit_ts > prev:
            out[p.instrument_key] = p.exit_ts
    return out


def _todays_trade_counts(
    positions: List[Position], now_ts: int
) -> Dict[str, int]:
    """Per-symbol count of positions (open or closed) entered today in IST."""
    today = _ist_session_date(now_ts)
    counts: Dict[str, int] = {}
    for p in positions:
        if _ist_session_date(p.entry_ts) == today:
            counts[p.instrument_key] = counts.get(p.instrument_key, 0) + 1
    return counts


def _cost_floor_inr(notional: float) -> float:
    """Minimum round-trip hard cost: 2 × max(₹20, 0.05% of notional) + STT."""
    brokerage_per_leg = max(20.0, 0.0005 * notional)
    stt = 0.00025 * notional  # 0.025% sell-side
    return brokerage_per_leg * 2 + stt


def decide(
    features_at_minute: pd.DataFrame,
    model: ModelArtifact,
    open_positions: List[Position],
    config: EngineConfig,
    now_ts: int,
    closed_positions: Optional[List[Position]] = None,
    sentiment_scores: Optional[Dict[str, float]] = None,
    allow_longs: bool = True,
    allow_shorts: bool = True,
) -> EngineCycleResult:
    """One decision cycle.

    features_at_minute: one row per candidate symbol with columns
        ['instrument_key', 'close', *FEATURE_COLUMNS]. Caller has already
        dropped rows with NaN features.
    open_positions: every currently open Position (long or short).
    Returns intents + per-skip-reason counters for the daily report.
    """
    result = EngineCycleResult()

    if is_forced_exit(now_ts, config):
        # End-of-day: emit exit intents for every open position. No new entries.
        for pos in open_positions:
            # Exit side is opposite of position side, but we emit by Position so
            # the executor can size correctly.
            result.add(
                OrderIntent(
                    instrument_key=pos.instrument_key,
                    side=pos.side,  # tagged with the existing side; executor closes
                    qty=pos.qty,
                    reason="exit_eod",
                    predicted_return=0.0,
                )
            )
        return result

    if not is_entry_window(now_ts, config):
        result.skip("outside_entry_window")
        return result

    if features_at_minute.empty:
        result.skip("no_candidates")
        return result

    available_slots = config.max_concurrent_positions - len(open_positions)
    if available_slots <= 0:
        result.skip("max_concurrent_reached")
        return result

    held_keys = {p.instrument_key for p in open_positions}
    # Exclude symbols we already hold (no pyramiding in v1).
    candidates = features_at_minute[~features_at_minute["instrument_key"].isin(held_keys)].copy()
    if candidates.empty:
        result.skip("all_candidates_already_held")
        return result

    # Anti-churn filters: cooldown + per-symbol daily cap. Both consult the day's
    # closed positions (passed in by the runner / live engine).
    closed = closed_positions or []
    cooldown_seconds = config.cooldown_minutes * 60
    last_exit_by_sym = _last_exit_ts_by_symbol(closed)
    trades_today_by_sym = _todays_trade_counts(closed + open_positions, now_ts)

    def _passes_anti_churn(sym: str) -> Optional[str]:
        """Return None if the symbol can be entered; else a skip-reason string."""
        last_exit = last_exit_by_sym.get(sym)
        if last_exit is not None and now_ts - last_exit < cooldown_seconds:
            return "cooldown_active"
        if trades_today_by_sym.get(sym, 0) >= config.max_trades_per_symbol_per_day:
            return "symbol_daily_cap_reached"
        return None

    keep = []
    for _, row in candidates.iterrows():
        skip_reason = _passes_anti_churn(str(row["instrument_key"]))
        if skip_reason:
            result.skip(skip_reason)
        else:
            keep.append(row)
    if not keep:
        return result
    candidates = pd.DataFrame(keep).reset_index(drop=True)

    # Score
    predictions = score(model, candidates)
    candidates["pred"] = predictions

    # Edge threshold
    eligible_long = candidates[candidates["pred"] >= config.min_predicted_edge].copy()
    eligible_short = candidates[candidates["pred"] <= -config.min_predicted_edge].copy()
    skipped_below_threshold = len(candidates) - len(eligible_long) - len(eligible_short)
    if skipped_below_threshold > 0:
        result.skipped_reasons["below_edge_threshold"] = skipped_below_threshold

    eligible_long = eligible_long.sort_values("pred", ascending=False).head(config.top_k_long)
    eligible_short = eligible_short.sort_values("pred", ascending=True).head(config.top_k_short)

    sent = sentiment_scores or {}
    veto = config.sentiment_veto_threshold

    if not allow_longs:
        result.skip("regime_bearish_longs_suppressed")
        eligible_long = eligible_long.iloc[0:0]
    if not allow_shorts:
        result.skip("regime_bullish_shorts_suppressed")
        eligible_short = eligible_short.iloc[0:0]

    longs_taken = 0
    shorts_taken = 0
    for _, row in eligible_long.iterrows():
        if longs_taken + shorts_taken >= available_slots:
            result.skip("max_concurrent_reached")
            break
        sym = str(row["instrument_key"])
        if veto > 0 and sent.get(sym, 0.0) < -veto:
            result.skip("sentiment_bearish_veto")
            continue
        qty = _qty_for_notional(config.capital_inr, float(row["close"]))
        if qty == 0:
            result.skip("qty_zero")
            continue
        result.add(
            OrderIntent(
                instrument_key=sym,
                side="short" if config.invert_signals else "long",
                qty=qty,
                reason="long_top_pick",
                predicted_return=float(row["pred"]),
            )
        )
        longs_taken += 1

    for _, row in eligible_short.iterrows():
        if longs_taken + shorts_taken >= available_slots:
            result.skip("max_concurrent_reached")
            break
        sym = str(row["instrument_key"])
        if veto > 0 and sent.get(sym, 0.0) > veto:
            result.skip("sentiment_bullish_veto")
            continue
        qty = _qty_for_notional(config.capital_inr, float(row["close"]))
        if qty == 0:
            result.skip("qty_zero")
            continue
        result.add(
            OrderIntent(
                instrument_key=sym,
                side="long" if config.invert_signals else "short",
                qty=qty,
                reason="short_bottom_pick",
                predicted_return=float(row["pred"]),
            )
        )
        shorts_taken += 1

    return result


def compute_stop_target(entry_price: float, side: Side, config: EngineConfig) -> tuple[float, float]:
    """Symmetric SL/TP placement. The executor passes these to the new Position."""
    if side == "long":
        return entry_price * (1 - config.stop_loss_pct), entry_price * (1 + config.target_pct)
    return entry_price * (1 + config.stop_loss_pct), entry_price * (1 - config.target_pct)
