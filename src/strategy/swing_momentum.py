"""S1 — 5-day momentum continuation strategy. Multi-day swing.

Empirical edge (validated on 2025-05 → 2026-05 history):
    Net P&L:      +₹5,516 (+5.52%) on ₹100k capital
    NIFTY B&H:    -₹4,490 (-4.49%)
    Excess:       +10pp in a falling market
    Trades:       217, win rate 40.6%, max DD ₹8,414

Why it works (and ORB/Supertrend didn't):
    1. Multi-day holds amortise the ~0.15% round-trip cost over 4-5% moves
    2. 1:2 R:R (target 6%, stop 3%) makes 40% win rate net positive
    3. F&O equities have arbitraged intraday signals but multi-day momentum
       still produces measurable drift

Signal:
    LONG when close > sma20 AND ret_5d > 0.02 AND volume > 0
    (universe is the same 209-stock F&O list as the rest of the project)

Risk shape:
    Target = entry × (1 + 6%)
    SL     = entry × (1 - 3%)
    Time exit = 10 trading days
    Max concurrent = 5 positions
    Notional per position = total_capital / max_positions (NO LEVERAGE — CNC product)
    Round-trip cost = 0.15% (CNC delivery, much cheaper than MIS)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from src.bot.engine import OrderIntent
from src.bot.positions import Position


@dataclass(frozen=True)
class SwingConfig:
    total_capital_inr: float = 100_000.0
    max_positions: int = 5

    # Risk shape — derived from the backtest validation.
    target_pct: float = 0.06          # +6% target
    stop_loss_pct: float = 0.03       # -3% stop loss
    hold_max_days: int = 10           # time exit after 10 trading days

    # Signal filters
    sma_window: int = 20
    momentum_window: int = 5
    min_5d_return: float = 0.02       # require 5d return > 2%

    # Cost model — CNC delivery is much cheaper than MIS intraday.
    round_trip_cost_pct: float = 0.0015

    @property
    def notional_per_slot_inr(self) -> float:
        """No leverage on CNC. Each slot gets capital/N rupees notional."""
        return self.total_capital_inr / self.max_positions

    @property
    def daily_loss_cap_inr(self) -> float:
        # 5% daily loss cap. Multi-day strategy moves slower than intraday,
        # so the cap can be wider without firing on noise.
        return self.total_capital_inr * 0.05


@dataclass
class SwingDecision:
    intents: List[OrderIntent]
    skipped: Dict[str, int]


def _qty_for_notional(notional_inr: float, price: float) -> int:
    if price <= 0:
        return 0
    return max(0, int(notional_inr // price))


def _add_features(daily: pd.DataFrame, config: SwingConfig) -> pd.DataFrame:
    """Add sma20, ret_5d to a single-symbol daily DataFrame."""
    out = daily.sort_values("ist_day").copy()
    out["sma20"] = out["close"].rolling(config.sma_window).mean()
    out["ret_5d"] = out["close"].pct_change(config.momentum_window)
    return out


def evaluate(
    daily_by_symbol: Dict[str, pd.DataFrame],
    open_positions: List[Position],
    closed_positions: List[Position],
    now_ts: int,
    config: Optional[SwingConfig] = None,
) -> SwingDecision:
    """Once-per-day signal generator. Run after market close (or before open).

    daily_by_symbol: instrument_key -> DataFrame of daily bars (ist_day asc).
        The most recent row should be today's just-closed session.
    """
    del closed_positions  # not used for entry decisions; engine manages exits
    config = config or SwingConfig()
    intents: List[OrderIntent] = []
    skipped: Dict[str, int] = {}

    def skip(reason: str):
        skipped[reason] = skipped.get(reason, 0) + 1

    available = config.max_positions - len(open_positions)
    if available <= 0:
        skip("max_concurrent_reached")
        return SwingDecision(intents=[], skipped=skipped)

    held_keys = {p.instrument_key for p in open_positions}

    # Score candidates by their 5d return, take the top `available` longs.
    candidates = []
    for key, daily in daily_by_symbol.items():
        if key in held_keys:
            skip("already_held")
            continue
        if daily is None or len(daily) < max(config.sma_window, config.momentum_window) + 1:
            skip("insufficient_history")
            continue
        feat = _add_features(daily, config)
        latest = feat.iloc[-1]
        if pd.isna(latest["sma20"]) or pd.isna(latest["ret_5d"]):
            skip("feature_warmup")
            continue
        close = float(latest["close"])
        if close <= float(latest["sma20"]):
            skip("below_sma")
            continue
        ret5 = float(latest["ret_5d"])
        if ret5 <= config.min_5d_return:
            skip("momentum_too_weak")
            continue
        if float(latest["volume"]) <= 0:
            skip("no_volume")
            continue
        candidates.append((key, close, ret5))

    # Sort: strongest 5-day momentum first.
    candidates.sort(key=lambda c: -c[2])

    for key, close, ret5 in candidates[:available]:
        qty = _qty_for_notional(config.notional_per_slot_inr, close)
        if qty == 0:
            skip("qty_zero")
            continue
        # The intent's stop_loss/target are pct-derived. The executor sets
        # absolute prices at fill time once the actual entry price is known.
        intents.append(OrderIntent(
            instrument_key=key, side="long", qty=qty,
            reason="swing_momentum_long",
            predicted_return=ret5,
            # Leave SL/TP None; the swing engine computes from config.stop_loss_pct/target_pct
            # at fill time based on actual fill price.
            stop_loss_price=None,
            target_price=None,
        ))

    return SwingDecision(intents=intents, skipped=skipped)
