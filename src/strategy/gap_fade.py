"""Gap Fade strategy.

Statistical regression on opening gaps. Stocks that open meaningfully away
from yesterday's close tend to fill that gap during the morning session. We
fade — go AGAINST the gap — once a small retracement confirms the move's
running out of steam.

Entry conditions (computed at every 5-min boundary in the entry window):
    Gap-down LONG : gap_pct <= -MIN_GAP_PCT and current_close has retraced
                    UP by RETRACEMENT_PCT from today's session low
    Gap-up   SHORT: gap_pct >= +MIN_GAP_PCT and current_close has retraced
                    DOWN by RETRACEMENT_PCT from today's session high

Exits:
    Target = previous session's close (fill the gap)
    SL     = today's session low - small buffer  (for long; mirror for short)
    EOD    = 14:30 IST via the broader engine

Filters:
    - gap magnitude must be at least MIN_GAP_PCT and at most MAX_GAP_PCT
      (too small = noise, too big = likely real news, don't fight)
    - one entry per stock per session
    - entry window 09:30 - 10:30 IST
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from src.bot.engine import OrderIntent
from src.bot.positions import Position
from src.features.technical import IST_OFFSET_SECONDS, SECONDS_PER_DAY
from src.strategy.rule_based import (
    StrategyDecision,
    _ist_minute_of_day,
    _qty_for_buying_power,
    RuleBasedConfig,
)


ENTRY_OPEN_MINUTE_IST = 9 * 60 + 30        # 09:30 IST — first valid fade entry
ENTRY_CLOSE_MINUTE_IST = 10 * 60 + 30       # 10:30 IST — last
MIN_GAP_PCT = 0.01                          # 1% gap minimum to attempt fade
MAX_GAP_PCT = 0.05                          # 5% — bigger than this is usually real news
RETRACEMENT_PCT = 0.003                     # 0.3% off today's extreme = "running out of steam"
SL_BUFFER_PCT = 0.001                       # 0.1% beyond today's high/low for SL
ENTRY_BUFFER_PCT = 0.0005


def _ist_session_date(ts: int) -> int:
    return (ts + IST_OFFSET_SECONDS) // SECONDS_PER_DAY


@dataclass
class _SessionContext:
    prev_close: float
    today_open: float
    today_high: float
    today_low: float
    gap_pct: float  # (today_open - prev_close) / prev_close


def _build_session_context(bars_5m: pd.DataFrame, session_date: int) -> Optional[_SessionContext]:
    """Today's open/high/low + previous session's close, in 5-min bars."""
    if bars_5m.empty:
        return None
    today = bars_5m[bars_5m["minute_ts"].apply(_ist_session_date) == session_date].sort_values("minute_ts")
    prior = bars_5m[bars_5m["minute_ts"].apply(_ist_session_date) < session_date].sort_values("minute_ts")
    if today.empty or prior.empty:
        return None
    prev_close = float(prior.iloc[-1]["close"])
    today_open = float(today.iloc[0]["open"])
    if prev_close <= 0:
        return None
    return _SessionContext(
        prev_close=prev_close,
        today_open=today_open,
        today_high=float(today["high"].max()),
        today_low=float(today["low"].min()),
        gap_pct=(today_open - prev_close) / prev_close,
    )


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
        result.skip("outside_gap_entry_window")
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

        ctx = _build_session_context(bars, today)
        if ctx is None:
            result.skip("session_context_unavailable")
            continue

        gap = ctx.gap_pct
        if abs(gap) < MIN_GAP_PCT:
            result.skip("gap_too_small")
            continue
        if abs(gap) > MAX_GAP_PCT:
            result.skip("gap_too_large_likely_news")
            continue

        # Latest 5-min bar's close — what we'd enter against.
        today_bars = bars[bars["minute_ts"].apply(_ist_session_date) == today].sort_values("minute_ts")
        close_now = float(today_bars.iloc[-1]["close"])

        # Gap DOWN: fade by going LONG once price retraces up from today's low.
        if gap <= -MIN_GAP_PCT:
            retracement = (close_now - ctx.today_low) / ctx.today_low if ctx.today_low > 0 else 0
            if retracement < RETRACEMENT_PCT:
                result.skip("retracement_not_yet")
                continue
            entry_limit = close_now * (1.0 + ENTRY_BUFFER_PCT)
            sl = ctx.today_low * (1.0 - SL_BUFFER_PCT)
            target = ctx.prev_close
            if target <= entry_limit:
                result.skip("target_already_passed")
                continue
            risk_per_share = entry_limit - sl
            if risk_per_share <= 0:
                result.skip("non_positive_risk")
                continue
            qty = _qty_for_buying_power(config.buying_power_per_stock_inr, entry_limit)
            if qty == 0:
                result.skip("qty_zero")
                continue
            result.intents.append(OrderIntent(
                instrument_key=key, side="long", qty=qty,
                reason="gap_fade_long",
                predicted_return=0.0,
                stop_loss_price=sl,
                target_price=target,
            ))
            available_slots -= 1

        # Gap UP: fade by going SHORT once price retraces down from today's high.
        else:  # gap >= MIN_GAP_PCT (positive)
            retracement = (ctx.today_high - close_now) / ctx.today_high if ctx.today_high > 0 else 0
            if retracement < RETRACEMENT_PCT:
                result.skip("retracement_not_yet")
                continue
            entry_limit = close_now * (1.0 - ENTRY_BUFFER_PCT)
            sl = ctx.today_high * (1.0 + SL_BUFFER_PCT)
            target = ctx.prev_close
            if target >= entry_limit:
                result.skip("target_already_passed")
                continue
            risk_per_share = sl - entry_limit
            if risk_per_share <= 0:
                result.skip("non_positive_risk")
                continue
            qty = _qty_for_buying_power(config.buying_power_per_stock_inr, entry_limit)
            if qty == 0:
                result.skip("qty_zero")
                continue
            result.intents.append(OrderIntent(
                instrument_key=key, side="short", qty=qty,
                reason="gap_fade_short",
                predicted_return=0.0,
                stop_loss_price=sl,
                target_price=target,
            ))
            available_slots -= 1

    return result
