"""Paper executor — opens and closes positions, persists each transition to DB.

Mirrors src.backtest.executor but with two key differences:
    1. Fills happen at the *current LTP* (the freshest quote we have), not at
       the next bar's open. Live, that bar hasn't happened yet.
    2. Every open and close writes to paper_positions immediately so a bot
       crash doesn't lose positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.bot.costs import round_trip_cost_at_price
from src.bot.engine import EngineConfig, OrderIntent, compute_stop_target
from src.bot.positions import Position
from src.paper.persistence import insert_open_position, mark_position_closed
from src.utils.logger import logger


@dataclass
class PaperExecutor:
    config: EngineConfig
    slippage_bps: float = 5.0
    model_name: str = "v1"

    async def fill_intent(
        self,
        intent: OrderIntent,
        fill_ts: int,
        last_quote_price: float,
        entry_sentiment_score: Optional[float] = None,
    ) -> Position:
        """Convert an OrderIntent into an open Position. Slippage applied against us."""
        if intent.side == "long":
            fill_price = last_quote_price * (1 + self.slippage_bps / 10_000.0)
        else:
            fill_price = last_quote_price * (1 - self.slippage_bps / 10_000.0)
        sl, tp = compute_stop_target(fill_price, intent.side, self.config)
        pos = Position(
            instrument_key=intent.instrument_key,
            side=intent.side,
            qty=intent.qty,
            entry_ts=fill_ts,
            entry_price=fill_price,
            stop_loss_price=sl,
            target_price=tp,
        )
        await insert_open_position(
            pos,
            predicted_return=intent.predicted_return,
            model_name=self.model_name,
            entry_sentiment_score=entry_sentiment_score,
        )
        logger.info(
            f"PAPER ENTRY {pos.side} {pos.qty} {pos.instrument_key} @ {pos.entry_price:.2f} "
            f"(sl={pos.stop_loss_price:.2f} tp={pos.target_price:.2f} pred={intent.predicted_return:+.4f})"
        )
        return pos

    async def close_position(
        self,
        pos: Position,
        exit_ts: int,
        exit_price: float,
        reason: str,
    ) -> None:
        """Close an open position. Slippage applied against us at exit."""
        if pos.side == "long":
            adj_exit = exit_price * (1 - self.slippage_bps / 10_000.0)
        else:
            adj_exit = exit_price * (1 + self.slippage_bps / 10_000.0)
        costs = round_trip_cost_at_price(
            qty=pos.qty,
            entry_price=pos.entry_price,
            exit_price=adj_exit,
            side=pos.side,
        )
        pos.close(exit_ts=exit_ts, exit_price=adj_exit, reason=reason, costs_inr=costs.total)
        await mark_position_closed(pos)
        logger.info(
            f"PAPER EXIT  {pos.side} {pos.qty} {pos.instrument_key} @ {pos.exit_price:.2f} "
            f"reason={reason} pnl={pos.realised_pnl_inr:+.2f}"
        )
