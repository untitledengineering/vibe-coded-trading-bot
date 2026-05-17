"""Opening Range Breakout strategy.

For each session, the OPENING RANGE is the first OR_DURATION_MINUTES of the
trading day (default 30 min, i.e. the 09:15–09:45 IST window). We mark:
    range_high = max(high) of those bars
    range_low  = min(low)  of those bars
    range_avg_volume = mean volume of those bars

During the ENTRY window (09:45–11:00 IST default), on each just-closed 5-min
bar we check:

    LONG  when close > range_high AND current_volume > 1.5 × range_avg_volume
    SHORT when close < range_low  AND current_volume > 1.5 × range_avg_volume

Exit shape:
    SL     = the opposite side of the range
    Target = entry ± 1× range_height (R:R 1:1; ORB rarely sustains the rare 1:2)
    EOD    = 14:30 IST per the broader config

Filters:
    - range must be at least MIN_RANGE_PCT of close (else it's noise)
    - range must not exceed MAX_RANGE_PCT (else already moved a lot)
    - no entries after ENTRY_CLOSE_MINUTE_IST
    - one entry per stock per session (the engine enforces via closed_today)
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


# ORB-specific knobs. Independent of RuleBasedConfig so we can tune without
# breaking the trend-follower (which is still committed in the repo).
OR_DURATION_MINUTES = 30                  # opening range = first 30 min = six 5-min bars
ENTRY_OPEN_MINUTE_IST = 9 * 60 + 45        # 09:45 IST — first valid entry minute
ENTRY_CLOSE_MINUTE_IST = 11 * 60           # 11:00 IST — last valid entry minute
MIN_RANGE_PCT = 0.003                      # 0.3% — anything smaller is bid/ask noise
MAX_RANGE_PCT = 0.02                       # 2.0% — already moved too much
VOLUME_CONFIRM_MULT = 1.5                  # current bar must clear 1.5x avg of OR bars
ENTRY_BUFFER_PCT = 0.0005                  # 0.05% above breakout / below breakdown
TARGET_RR_MULTIPLIER = 2.0                 # target = entry + 2.0 × risk. Indices need
                                            # R:R >= 1.5 to overcome the asymmetric SL
                                            # hits we see on intraday data (40% win
                                            # rate * 1R - 60% * 1R = -20%; same WR with
                                            # 2R target = +20% EV before costs).


@dataclass
class _OpeningRange:
    high: float
    low: float
    avg_volume: float

    @property
    def height(self) -> float:
        return self.high - self.low


def _ist_session_date(ts: int) -> int:
    return (ts + IST_OFFSET_SECONDS) // SECONDS_PER_DAY


def _compute_opening_range(bars_5m: pd.DataFrame, session_date: int) -> Optional[_OpeningRange]:
    """The first OR_DURATION_MINUTES of `session_date` from the 5-min bars."""
    if bars_5m.empty:
        return None
    n_bars = OR_DURATION_MINUTES // 5
    same_session = bars_5m[
        bars_5m["minute_ts"].apply(_ist_session_date) == session_date
    ].sort_values("minute_ts")
    if len(same_session) < n_bars:
        return None
    head = same_session.head(n_bars)
    return _OpeningRange(
        high=float(head["high"].max()),
        low=float(head["low"].min()),
        avg_volume=float(head["volume"].mean()) if head["volume"].sum() > 0 else 0.0,
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
    """One decision cycle for ORB. Called at every 5-min boundary."""
    config = config or RuleBasedConfig()
    result = StrategyDecision()
    today = _ist_session_date(now_ts)

    if not _in_entry_window(now_ts):
        result.skip("outside_orb_entry_window")
        return result

    available_slots = config.max_positions - len(open_positions)
    if available_slots <= 0:
        result.skip("max_concurrent_reached")
        return result

    held_keys = {p.instrument_key for p in open_positions}
    # One ORB entry per stock per day: if we already traded a symbol today
    # (open OR closed), skip it.
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

        # The current 5-min bar must be in today's session and AFTER the OR window.
        today_bars = bars[bars["minute_ts"].apply(_ist_session_date) == today].sort_values("minute_ts")
        if len(today_bars) <= (OR_DURATION_MINUTES // 5):
            result.skip("no_post_range_bars")
            continue
        latest = today_bars.iloc[-1]
        close_now = float(latest["close"])
        volume_now = float(latest["volume"]) if latest["volume"] is not None else 0.0

        range_pct = opening_range.height / close_now if close_now > 0 else 0.0
        if range_pct < MIN_RANGE_PCT:
            result.skip("range_too_small")
            continue
        if range_pct > MAX_RANGE_PCT:
            result.skip("range_too_wide")
            continue

        if (
            opening_range.avg_volume > 0
            and volume_now < VOLUME_CONFIRM_MULT * opening_range.avg_volume
        ):
            result.skip("volume_unconfirmed")
            continue

        # LONG breakout above the range high.
        if close_now > opening_range.high:
            entry_limit = close_now * (1.0 + ENTRY_BUFFER_PCT)
            sl = opening_range.low
            risk_per_share = entry_limit - sl
            if risk_per_share <= 0:
                result.skip("non_positive_risk")
                continue
            target = entry_limit + TARGET_RR_MULTIPLIER * risk_per_share
            qty = _qty_for_buying_power(config.buying_power_per_stock_inr, entry_limit)
            if qty == 0:
                result.skip("qty_zero")
                continue
            result.intents.append(OrderIntent(
                instrument_key=key, side="long", qty=qty,
                reason="orb_long_breakout",
                predicted_return=0.0,
                stop_loss_price=sl,
                target_price=target,
            ))
            available_slots -= 1

        # SHORT breakdown below the range low.
        elif close_now < opening_range.low:
            entry_limit = close_now * (1.0 - ENTRY_BUFFER_PCT)
            sl = opening_range.high
            risk_per_share = sl - entry_limit
            if risk_per_share <= 0:
                result.skip("non_positive_risk")
                continue
            target = entry_limit - TARGET_RR_MULTIPLIER * risk_per_share
            qty = _qty_for_buying_power(config.buying_power_per_stock_inr, entry_limit)
            if qty == 0:
                result.skip("qty_zero")
                continue
            result.intents.append(OrderIntent(
                instrument_key=key, side="short", qty=qty,
                reason="orb_short_breakdown",
                predicted_return=0.0,
                stop_loss_price=sl,
                target_price=target,
            ))
            available_slots -= 1
        else:
            result.skip("inside_range")

    return result
