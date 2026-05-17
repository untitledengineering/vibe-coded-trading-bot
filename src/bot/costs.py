"""Upstox MIS (intraday cash equity) cost model.

Sources cross-checked against the Upstox pricing page and the SEBI/exchange
charge schedules. As of mid-2026:

    Brokerage:           min(₹20, 0.05% of executed value) per leg
    STT:                 0.025% of sell value (sell side only)
    Exchange transaction: 0.00345% per leg (NSE equity cash)
    SEBI charges:        0.0001% per leg
    Stamp duty:          0.003% of buy value (buy side only)
    GST:                 18% on (brokerage + exchange + SEBI)

All inputs are absolute INR values. Output is INR.
"""

from __future__ import annotations

from dataclasses import dataclass

BROKERAGE_FLAT_INR = 20.0
BROKERAGE_PCT = 0.0005  # 0.05% per leg
STT_PCT_SELL = 0.00025  # 0.025% of sell value
EXCHANGE_PCT = 0.0000345  # 0.00345% per leg
SEBI_PCT = 0.000001  # 0.0001% per leg
STAMP_PCT_BUY = 0.00003  # 0.003% of buy value
GST_PCT = 0.18


@dataclass(frozen=True)
class CostBreakdown:
    brokerage: float
    stt: float
    exchange: float
    sebi: float
    stamp: float
    gst: float

    @property
    def total(self) -> float:
        return self.brokerage + self.stt + self.exchange + self.sebi + self.stamp + self.gst


def leg_brokerage(notional: float) -> float:
    return min(BROKERAGE_FLAT_INR, BROKERAGE_PCT * notional)


def round_trip_cost(buy_value: float, sell_value: float) -> CostBreakdown:
    """Cost of opening and closing one MIS position.

    buy_value  — INR value of the buy leg (entry for long, exit for short)
    sell_value — INR value of the sell leg (exit for long, entry for short)

    The model is symmetric in side: shorts pay STT on the sell-side entry and
    stamp on the buy-side exit, same total."""
    brokerage = leg_brokerage(buy_value) + leg_brokerage(sell_value)
    stt = STT_PCT_SELL * sell_value
    exchange = EXCHANGE_PCT * (buy_value + sell_value)
    sebi = SEBI_PCT * (buy_value + sell_value)
    stamp = STAMP_PCT_BUY * buy_value
    gst = GST_PCT * (brokerage + exchange + sebi)
    return CostBreakdown(
        brokerage=brokerage,
        stt=stt,
        exchange=exchange,
        sebi=sebi,
        stamp=stamp,
        gst=gst,
    )


def round_trip_cost_at_price(qty: int, entry_price: float, exit_price: float, side: str) -> CostBreakdown:
    """Convenience wrapper. side ∈ {'long', 'short'}."""
    if side == "long":
        return round_trip_cost(buy_value=qty * entry_price, sell_value=qty * exit_price)
    if side == "short":
        return round_trip_cost(buy_value=qty * exit_price, sell_value=qty * entry_price)
    raise ValueError(f"side must be 'long' or 'short', got {side!r}")


def slippage_inr(notional: float, slippage_bps: float = 5.0) -> float:
    """Notional × (bps / 10000) per leg. Default 5 bps each side is a conservative
    estimate for liquid Indian F&O equity at midcap-and-above sizes."""
    return notional * (slippage_bps / 10_000.0)
