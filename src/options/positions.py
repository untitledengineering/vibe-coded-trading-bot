"""Spread position model.

A credit spread is a defined-risk options position: sell one option (the short
leg) and simultaneously buy a further-OTM option of the same type (the long
leg, providing the hedge). The net premium received at entry is the credit.

Two spread types in v1:
    bull_put_spread:  sell PUT at strike A, buy PUT at strike B (B < A)
                      Bullish/neutral bias — profits if spot stays above A
    bear_call_spread: sell CALL at strike A, buy CALL at strike B (B > A)
                      Bearish/neutral bias — profits if spot stays below A

Max profit = credit per lot × lot_size × qty_lots         (capped at the credit)
Max loss   = (spread_width − credit_per_lot) × lot_size × qty_lots
Margin     = roughly the max-loss above (SPAN+ELM is similar in practice)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

SpreadType = Literal["bull_put_spread", "bear_call_spread"]


@dataclass
class SpreadPosition:
    underlying: str                  # e.g., 'NSE_INDEX|Nifty 50'
    spread_type: SpreadType
    expiry_date: str                  # 'YYYY-MM-DD'
    entry_ts: int                     # epoch seconds

    short_leg_key: str                # option contract instrument_key
    short_strike: float
    short_entry_price: float

    long_leg_key: str
    long_strike: float
    long_entry_price: float

    qty_lots: int                     # number of LOTS (each lot = lot_size units)
    lot_size: int

    # Risk knobs
    stop_loss_pct_of_credit: float = 2.0   # exit when mark loss > 2x credit received
    target_pct_of_credit: float = 0.5      # exit when 50% of credit captured

    # Exit fields populated when the spread closes
    exit_ts: Optional[int] = None
    short_exit_price: Optional[float] = None
    long_exit_price: Optional[float] = None
    realised_pnl_inr: Optional[float] = None
    exit_reason: Optional[str] = None

    # ---- Derived metrics ----

    @property
    def credit_received_per_lot(self) -> float:
        """Net premium received at entry (in points). Always positive for a credit spread."""
        return self.short_entry_price - self.long_entry_price

    @property
    def spread_width(self) -> float:
        """Strike-to-strike distance in points."""
        return abs(self.short_strike - self.long_strike)

    @property
    def max_loss_per_lot(self) -> float:
        """Worst-case loss in points per lot (if both legs go fully against us)."""
        return max(0.0, self.spread_width - self.credit_received_per_lot)

    @property
    def max_loss_inr(self) -> float:
        return self.max_loss_per_lot * self.lot_size * self.qty_lots

    @property
    def max_profit_inr(self) -> float:
        return self.credit_received_per_lot * self.lot_size * self.qty_lots

    @property
    def is_open(self) -> bool:
        return self.exit_ts is None

    def unrealised_pnl_inr(self, short_mark: float, long_mark: float) -> float:
        """Mark-to-market P&L. We sold short, so we want short price to FALL.
        We bought long, so we want long price to RISE. Net per lot = received
        − current_cost_to_close."""
        per_lot_pnl = (self.short_entry_price - short_mark) - (self.long_entry_price - long_mark)
        return per_lot_pnl * self.lot_size * self.qty_lots

    def close(
        self,
        exit_ts: int,
        short_exit_price: float,
        long_exit_price: float,
        reason: str,
        costs_inr: float = 0.0,
    ) -> None:
        self.exit_ts = exit_ts
        self.short_exit_price = short_exit_price
        self.long_exit_price = long_exit_price
        self.exit_reason = reason
        gross = self.unrealised_pnl_inr(short_exit_price, long_exit_price)
        self.realised_pnl_inr = gross - costs_inr


def estimate_spread_margin_inr(spread: SpreadPosition) -> float:
    """Conservative margin estimate for SPAN+ELM on a defined-risk spread.

    For a hedged short option (both legs same expiry, same type), Indian
    brokers typically require roughly the max-loss as margin since that's the
    bounded downside. Real SPAN can be slightly less (~80-95% of max loss for
    deep-OTM spreads), but conservative estimate keeps us safe."""
    return spread.max_loss_inr
