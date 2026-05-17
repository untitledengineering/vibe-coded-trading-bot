"""Position tracking. The same Position type is used by backtest and live
paper. Keeping it pure data + a few pure helpers avoids any divergence in how
P&L is computed between the two."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Side = Literal["long", "short"]


@dataclass
class Position:
    instrument_key: str
    side: Side
    qty: int
    entry_ts: int             # epoch seconds
    entry_price: float
    stop_loss_price: float    # absolute price at which we exit
    target_price: float       # absolute price at which we take profit
    exit_ts: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "stop_loss" | "target" | "eod" | "kill_switch"
    realised_pnl_inr: Optional[float] = None  # filled at close (after costs)
    # Trailing-to-cost state. Set True after we've moved the SL to entry on
    # +1R move in our favour. Once locked, we never trail back.
    breakeven_locked: bool = False
    # Entry rationale — populated from the DB when loaded, None when created in backtests.
    predicted_return: Optional[float] = None
    model_name: Optional[str] = None
    entry_sentiment_score: Optional[float] = None

    @property
    def is_open(self) -> bool:
        return self.exit_ts is None

    @property
    def notional(self) -> float:
        return self.qty * self.entry_price

    def unrealised_pnl_inr(self, mark_price: float) -> float:
        """Mark-to-market P&L in INR. Long = qty × (mark - entry); short flips sign."""
        delta = mark_price - self.entry_price
        if self.side == "short":
            delta = -delta
        return self.qty * delta

    def should_exit_at(self, high: float, low: float) -> Optional[str]:
        """Check the bar's high/low against SL/TP. Returns the exit reason if
        either trigger fired, else None. We assume the worse case for the trader
        (SL checked before TP) — this is the conservative backtest convention.
        """
        if self.side == "long":
            if low <= self.stop_loss_price:
                return "stop_loss"
            if high >= self.target_price:
                return "target"
        else:  # short
            if high >= self.stop_loss_price:
                return "stop_loss"
            if low <= self.target_price:
                return "target"
        return None

    def close(self, exit_ts: int, exit_price: float, reason: str, costs_inr: float = 0.0) -> None:
        self.exit_ts = exit_ts
        self.exit_price = exit_price
        self.exit_reason = reason
        gross = self.unrealised_pnl_inr(exit_price)
        self.realised_pnl_inr = gross - costs_inr

    def maybe_trail_to_breakeven(self, current_price: float) -> bool:
        """Move stop_loss_price to entry_price when the trade is +1R in favour.

        Used by the rule-based strategy to lock in a risk-free trade once the
        market has paid for the initial risk. Returns True if the SL was moved
        on this call (so the engine can persist + notify); False otherwise.

        Idempotent: once breakeven_locked is True, further calls are no-ops.
        """
        if self.breakeven_locked or self.exit_ts is not None:
            return False
        risk_per_share = abs(self.entry_price - self.stop_loss_price)
        if risk_per_share <= 0:
            return False
        if self.side == "long":
            profit_per_share = current_price - self.entry_price
        else:
            profit_per_share = self.entry_price - current_price
        if profit_per_share < risk_per_share:
            return False
        self.stop_loss_price = self.entry_price
        self.breakeven_locked = True
        return True
