"""Opening Range FADE — the inverse of ORB.

ORB-momentum bets that a break of the opening range continues. ORB-fade bets
that the break is exhaustion and will revert to the range. Same opening range
construction, same time window, OPPOSITE trade direction:

    SHORT when close > range_high  (fade the up-break, expect revert)
    LONG  when close < range_low   (fade the down-break, expect revert)

Targets and stops reflect the mean-reversion thesis:
    Target = range_midpoint   (revert most of the way back)
    SL     = small fixed buffer past the entry (so we don't sit through the
             actual breakout if we're wrong)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from src.bot.engine import OrderIntent
from src.bot.positions import Position
from src.features.technical import IST_OFFSET_SECONDS, SECONDS_PER_DAY
from src.strategy.orb import (
    ENTRY_CLOSE_MINUTE_IST,
    ENTRY_OPEN_MINUTE_IST,
    MAX_RANGE_PCT,
    MIN_RANGE_PCT,
    _compute_opening_range,
)
from src.strategy.rule_based import (
    StrategyDecision,
    _ist_minute_of_day,
    _qty_for_buying_power,
    RuleBasedConfig,
)


# Fade-specific risk knobs.
SL_BUFFER_PCT = 0.0025          # 0.25% beyond the breakout point — tight stop
TARGET_RR_MULTIPLIER = 1.5      # target = entry ± 1.5 × SL distance (the bias should
                                # be quick reversion to range midpoint, then further)


def _ist_session_date(ts: int) -> int:
    return (ts + IST_OFFSET_SECONDS) // SECONDS_PER_DAY


def _in_entry_window(now_ts: int) -> bool:
    minute = _ist_minute_of_day(now_ts)
    return ENTRY_OPEN_MINUTE_IST <= minute <= ENTRY_CLOSE_MINUTE_IST


def evaluate(
    bars_5m_by_symbol: Dict[str, pd.DataFrame],
    open_positions: List[Position],
    closed_today: List[Position],
    now_ts: int,
    config: Optional[RuleBasedConfig] = None,
) -> StrategyDecision:
    config = config or RuleBasedConfig()
    result = StrategyDecision()
    today = _ist_session_date(now_ts)

    if not _in_entry_window(now_ts):
        result.skip("outside_orb_fade_window")
        return result

    available_slots = config.max_positions - len(open_positions)
    if available_slots <= 0:
        result.skip("max_concurrent_reached")
        return result

    held_keys = {p.instrument_key for p in open_positions}
    traded_today_keys = held_keys | {
        p.instrument_key for p in closed_today
        if p.entry_ts and _ist_session_date(p.entry_ts) == today
    }

    for key, bars in bars_5m_by_symbol.items():
        if available_slots <= 0:
            break
        if key in traded_today_keys:
            result.skip("already_traded_today")
            continue
        if bars is None or bars.empty:
            result.skip("no_bars")
            continue

        opening_range = _compute_opening_range(bars, today)
        if opening_range is None:
            result.skip("opening_range_not_ready")
            continue

        today_bars = bars[bars["minute_ts"].apply(_ist_session_date) == today].sort_values("minute_ts")
        if len(today_bars) <= 6:
            result.skip("no_post_range_bars")
            continue
        latest = today_bars.iloc[-1]
        close_now = float(latest["close"])

        range_pct = opening_range.height / close_now if close_now > 0 else 0.0
        if range_pct < MIN_RANGE_PCT:
            result.skip("range_too_small")
            continue
        if range_pct > MAX_RANGE_PCT:
            result.skip("range_too_wide")
            continue

        midpoint = (opening_range.high + opening_range.low) / 2.0

        # Up-break -> SHORT (fade)
        if close_now > opening_range.high:
            entry_limit = close_now
            sl = close_now * (1.0 + SL_BUFFER_PCT)
            target = midpoint
            risk = sl - entry_limit
            reward = entry_limit - target
            if risk <= 0 or reward <= 0:
                result.skip("non_positive_risk_or_reward")
                continue
            # Only fade if the implied R:R is reasonable.
            if reward / risk < TARGET_RR_MULTIPLIER:
                result.skip("rr_below_threshold")
                continue
            qty = _qty_for_buying_power(config.buying_power_per_stock_inr, entry_limit)
            if qty == 0:
                result.skip("qty_zero")
                continue
            result.intents.append(OrderIntent(
                instrument_key=key, side="short", qty=qty,
                reason="orb_fade_short",
                predicted_return=0.0,
                stop_loss_price=sl,
                target_price=target,
            ))
            available_slots -= 1

        # Down-break -> LONG (fade)
        elif close_now < opening_range.low:
            entry_limit = close_now
            sl = close_now * (1.0 - SL_BUFFER_PCT)
            target = midpoint
            risk = entry_limit - sl
            reward = target - entry_limit
            if risk <= 0 or reward <= 0:
                result.skip("non_positive_risk_or_reward")
                continue
            if reward / risk < TARGET_RR_MULTIPLIER:
                result.skip("rr_below_threshold")
                continue
            qty = _qty_for_buying_power(config.buying_power_per_stock_inr, entry_limit)
            if qty == 0:
                result.skip("qty_zero")
                continue
            result.intents.append(OrderIntent(
                instrument_key=key, side="long", qty=qty,
                reason="orb_fade_long",
                predicted_return=0.0,
                stop_loss_price=sl,
                target_price=target,
            ))
            available_slots -= 1
        else:
            result.skip("inside_range")

    return result
