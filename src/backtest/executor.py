"""Backtest executor: turns OrderIntents into simulated fills.

Conventions:
    - Entries fill at the NEXT minute's open price (we cannot peek at the bar
      the signal was generated on).
    - Slippage is applied unfavourably: longs pay +X bps, shorts pay -X bps.
    - SL/TP exits fill at the level itself plus slippage. Conservative.
    - End-of-day exits fill at the 14:55 IST bar's close.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bot.costs import round_trip_cost_at_price
from src.bot.engine import EngineConfig, OrderIntent, compute_stop_target
from src.bot.positions import Position


@dataclass
class BacktestExecutor:
    config: EngineConfig
    slippage_bps: float = 5.0

    def open_position(
        self,
        intent: OrderIntent,
        fill_ts: int,
        bar_open_price: float,
    ) -> Position:
        # Slippage works against us at entry.
        if intent.side == "long":
            fill_price = bar_open_price * (1 + self.slippage_bps / 10_000.0)
        else:
            fill_price = bar_open_price * (1 - self.slippage_bps / 10_000.0)
        # Rule-based strategies pre-compute absolute SL/TP (e.g. prev-candle-low
        # for longs). ML-based strategies leave them None and fall back to the
        # config's percentage offsets from fill_price.
        if intent.stop_loss_price is not None and intent.target_price is not None:
            sl, tp = intent.stop_loss_price, intent.target_price
        else:
            sl, tp = compute_stop_target(fill_price, intent.side, self.config)
        return Position(
            instrument_key=intent.instrument_key,
            side=intent.side,
            qty=intent.qty,
            entry_ts=fill_ts,
            entry_price=fill_price,
            stop_loss_price=sl,
            target_price=tp,
        )

    def close_position(
        self,
        position: Position,
        exit_ts: int,
        exit_price: float,
        reason: str,
    ) -> None:
        # Slippage against us at exit too.
        if position.side == "long":
            adjusted_exit = exit_price * (1 - self.slippage_bps / 10_000.0)
        else:
            adjusted_exit = exit_price * (1 + self.slippage_bps / 10_000.0)
        costs = round_trip_cost_at_price(
            qty=position.qty,
            entry_price=position.entry_price,
            exit_price=adjusted_exit,
            side=position.side,
        )
        position.close(
            exit_ts=exit_ts,
            exit_price=adjusted_exit,
            reason=reason,
            costs_inr=costs.total,
        )
