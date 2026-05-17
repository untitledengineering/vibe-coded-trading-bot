"""VWAP + Supertrend(7,3) rule-based intraday strategy.

Faithful to the v0.2 spec:

  LONG  when current 5-min candle closes above VWAP AND Supertrend(7,3) is green
  SHORT when current 5-min candle closes below VWAP AND Supertrend(7,3) is red

Stop loss is the previous 5-min candle's low (for longs) or high (for shorts),
clamped so the |entry - SL| never exceeds 1% of entry. Target is exactly 2× the
SL distance (1:2 R:R). Trailing-to-breakeven on +1R is handled at the engine
level, not here — the strategy is stateless.

Position sizing: total_capital / max_positions per stock, scaled by leverage.
At ₹25k capital and 6 positions, each stock gets ~₹4,167 × 5 = ~₹20.8k buying
power, which is 8 shares of RELIANCE at ₹2,600 or 15 shares at ₹1,340.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src.bot.engine import EngineCycleResult, OrderIntent
from src.bot.positions import Position
from src.features.technical import IST_OFFSET_SECONDS, SECONDS_PER_DAY
from src.strategy.indicators import session_vwap, supertrend


# v0.2 universe: 6 sector representatives, 1 each.
# NOTE: post the 2025-2026 Tata Motors demerger, the original "TATAMOTORS" ticker
# split. We use TMPV (Tata Motors Passenger Vehicles, INE155A01022) which kept
# the original ISIN and is the cleanest "Tata Motors-like" name in F&O.
RULE_BASED_UNIVERSE = [
    {"instrument_key": "NSE_EQ|INE040A01034", "trading_symbol": "HDFCBANK", "sector": "Banking"},
    {"instrument_key": "NSE_EQ|INE009A01021", "trading_symbol": "INFY",     "sector": "IT"},
    {"instrument_key": "NSE_EQ|INE002A01018", "trading_symbol": "RELIANCE", "sector": "Energy"},
    {"instrument_key": "NSE_EQ|INE155A01022", "trading_symbol": "TMPV",     "sector": "Auto"},
    {"instrument_key": "NSE_EQ|INE154A01025", "trading_symbol": "ITC",      "sector": "FMCG"},
    {"instrument_key": "NSE_EQ|INE019A01038", "trading_symbol": "JSWSTEEL", "sector": "Metals"},
]
RULE_BASED_INSTRUMENT_KEYS = [u["instrument_key"] for u in RULE_BASED_UNIVERSE]
SECTOR_BY_KEY = {u["instrument_key"]: u["sector"] for u in RULE_BASED_UNIVERSE}


@dataclass(frozen=True)
class RuleBasedConfig:
    """Spec-driven config. All knobs are bound to the v0.2 strategy doc."""
    total_capital_inr: float = 25_000.0
    max_positions: int = 6
    leverage: float = 5.0

    entry_buffer_pct: float = 0.0005       # 0.05% above/below close on the limit order
    stop_loss_pct_cap: float = 0.01        # 1% max risk per trade
    risk_reward_ratio: float = 2.0         # 1:2

    # Trading window in IST minute-of-day.
    entry_window_open_minute_ist: int = 9 * 60 + 20        # 09:20
    entry_window_close_minute_ist: int = 14 * 60 + 30      # 14:30
    forced_exit_minute_ist: int = 15 * 60 + 15             # 15:15

    # Account-level kill switches.
    daily_loss_pct_cap: float = 0.02       # 2% of total_capital_inr
    consecutive_loss_halt: int = 3
    consec_loss_pause_minutes: int = 60

    # "Trending market" override threshold: if Nifty has moved this much by
    # 11:00 IST, drop the 1-per-sector rule. With this 6-stock universe each
    # already in a distinct sector, this is a forward-compat knob — no effect today.
    trending_market_pct_threshold: float = 0.015           # ±1.5%

    @property
    def capital_per_stock_inr(self) -> float:
        return self.total_capital_inr / self.max_positions

    @property
    def buying_power_per_stock_inr(self) -> float:
        return self.capital_per_stock_inr * self.leverage

    @property
    def daily_loss_cap_inr(self) -> float:
        return self.total_capital_inr * self.daily_loss_pct_cap


@dataclass
class StrategyDecision:
    intents: List[OrderIntent] = field(default_factory=list)
    skipped: Dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


def _ist_minute_of_day(ts: int) -> int:
    return ((ts + IST_OFFSET_SECONDS) % SECONDS_PER_DAY) // 60


def is_entry_window(now_ts: int, config: RuleBasedConfig) -> bool:
    m = _ist_minute_of_day(now_ts)
    return config.entry_window_open_minute_ist <= m <= config.entry_window_close_minute_ist


def is_forced_exit(now_ts: int, config: RuleBasedConfig) -> bool:
    return _ist_minute_of_day(now_ts) >= config.forced_exit_minute_ist


def _qty_for_buying_power(buying_power_inr: float, price: float) -> int:
    if price <= 0:
        return 0
    return max(0, int(buying_power_inr // price))


def _clamp_long_sl(prev_low: float, entry_close: float, cap_pct: float) -> float:
    """Long stop = prev candle low, but never further than cap_pct below entry."""
    cap_floor = entry_close * (1.0 - cap_pct)
    return max(prev_low, cap_floor)  # tighter (higher) SL wins


def _clamp_short_sl(prev_high: float, entry_close: float, cap_pct: float) -> float:
    """Short stop = prev candle high, but never further than cap_pct above entry."""
    cap_ceiling = entry_close * (1.0 + cap_pct)
    return min(prev_high, cap_ceiling)  # tighter (lower) SL wins


def evaluate(
    bars_5m_by_symbol: Dict[str, pd.DataFrame],
    open_positions: List[Position],
    closed_today: List[Position],
    now_ts: int,
    config: Optional[RuleBasedConfig] = None,
) -> StrategyDecision:
    """One decision cycle. Inputs are deliberately explicit — same function used
    by both the live paper engine and the backtester. `closed_today` is part of
    the contract so the caller's kill-switch logic (consecutive losses, daily
    loss cap) can use the same shape, even though the strategy itself doesn't
    consult it directly."""
    del closed_today  # intentional: kill-switch logic lives in the engine
    config = config or RuleBasedConfig()
    result = StrategyDecision()

    if is_forced_exit(now_ts, config):
        # Engine handles the actual squareoff; strategy emits no entries here.
        result.skip("forced_exit_window")
        return result

    if not is_entry_window(now_ts, config):
        result.skip("outside_entry_window")
        return result

    available_slots = config.max_positions - len(open_positions)
    if available_slots <= 0:
        result.skip("max_concurrent_reached")
        return result

    held_keys = {p.instrument_key for p in open_positions}

    for key, bars in bars_5m_by_symbol.items():
        if available_slots <= 0:
            break
        if key in held_keys:
            result.skip("already_held")
            continue
        if bars is None or len(bars) < 10:  # need at least Supertrend warmup
            result.skip("insufficient_bars")
            continue

        # Compute indicators on the symbol's 5-min bars.
        vwap = session_vwap(bars)
        st_line, st_dir = supertrend(bars["high"], bars["low"], bars["close"], period=7, multiplier=3.0)

        latest = bars.iloc[-1]
        prev = bars.iloc[-2]
        vwap_now = vwap.iloc[-1]
        dir_now = int(st_dir.iloc[-1])

        if pd.isna(vwap_now) or dir_now == 0:
            result.skip("indicator_warmup")
            continue

        close_now = float(latest["close"])
        # Long: close above VWAP AND Supertrend green.
        if close_now > vwap_now and dir_now == 1:
            entry_limit = close_now * (1.0 + config.entry_buffer_pct)
            sl = _clamp_long_sl(float(prev["low"]), close_now, config.stop_loss_pct_cap)
            risk_per_share = entry_limit - sl
            if risk_per_share <= 0:
                result.skip("non_positive_risk")
                continue
            target = entry_limit + config.risk_reward_ratio * risk_per_share
            qty = _qty_for_buying_power(config.buying_power_per_stock_inr, entry_limit)
            if qty == 0:
                result.skip("qty_zero")
                continue
            result.intents.append(OrderIntent(
                instrument_key=key, side="long", qty=qty,
                reason="vwap_supertrend_long",
                predicted_return=0.0,
                stop_loss_price=sl,
                target_price=target,
            ))
            available_slots -= 1
        # Short: close below VWAP AND Supertrend red.
        elif close_now < vwap_now and dir_now == -1:
            entry_limit = close_now * (1.0 - config.entry_buffer_pct)
            sl = _clamp_short_sl(float(prev["high"]), close_now, config.stop_loss_pct_cap)
            risk_per_share = sl - entry_limit
            if risk_per_share <= 0:
                result.skip("non_positive_risk")
                continue
            target = entry_limit - config.risk_reward_ratio * risk_per_share
            qty = _qty_for_buying_power(config.buying_power_per_stock_inr, entry_limit)
            if qty == 0:
                result.skip("qty_zero")
                continue
            result.intents.append(OrderIntent(
                instrument_key=key, side="short", qty=qty,
                reason="vwap_supertrend_short",
                predicted_return=0.0,
                stop_loss_price=sl,
                target_price=target,
            ))
            available_slots -= 1
        else:
            result.skip("no_signal")

    return result


def consecutive_losses(closed_today: List[Position]) -> int:
    """Tail-count of consecutive losing trades (by exit_ts order)."""
    sorted_closed = sorted(closed_today, key=lambda p: p.exit_ts or 0)
    count = 0
    for p in reversed(sorted_closed):
        if (p.realised_pnl_inr or 0.0) < 0:
            count += 1
        else:
            break
    return count


def to_engine_cycle_result(d: StrategyDecision) -> EngineCycleResult:
    """Adapter so the rule-based output can be consumed by the existing engine
    code paths that expect EngineCycleResult."""
    out = EngineCycleResult()
    for i in d.intents:
        out.intents.append(i)
    for k, v in d.skipped.items():
        out.skipped_reasons[k] = v
    return out
