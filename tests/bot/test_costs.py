"""Cost-model tests. The exact rupee figures come from the Upstox / SEBI charge schedule."""

import pytest

from src.bot.costs import (
    BROKERAGE_FLAT_INR,
    BROKERAGE_PCT,
    leg_brokerage,
    round_trip_cost,
    round_trip_cost_at_price,
    slippage_inr,
)


def test_leg_brokerage_caps_at_flat_for_large_notional():
    # 0.05% of ₹100,000 = ₹50 > flat ₹20 cap.
    assert leg_brokerage(100_000) == BROKERAGE_FLAT_INR


def test_leg_brokerage_pct_for_small_notional():
    # 0.05% of ₹10,000 = ₹5 < flat cap.
    assert leg_brokerage(10_000) == pytest.approx(5.0)


def test_round_trip_breakdown_on_25k_position():
    """Sanity-check actual numbers for a typical ₹25k MIS trade."""
    br = round_trip_cost(buy_value=25_000, sell_value=25_000)
    # Brokerage: 0.05% × 25k = ₹12.50 per leg × 2 legs = ₹25
    assert br.brokerage == pytest.approx(25.0)
    # STT 0.025% on sell only.
    assert br.stt == pytest.approx(6.25)
    # Exchange: 0.00345% × 50k.
    assert br.exchange == pytest.approx(0.0000345 * 50_000)
    # Stamp duty on buy only: 0.003% × 25k.
    assert br.stamp == pytest.approx(0.75)
    # GST: 18% × (brokerage + exchange + sebi).
    assert br.gst == pytest.approx(0.18 * (br.brokerage + br.exchange + br.sebi))
    # Total in the ~₹38 ballpark.
    assert 35 < br.total < 45


def test_round_trip_cost_at_price_long_and_short_symmetric():
    """A long round-tripped at flat prices costs the same as the symmetric short."""
    long_cost = round_trip_cost_at_price(qty=100, entry_price=250, exit_price=250, side="long")
    short_cost = round_trip_cost_at_price(qty=100, entry_price=250, exit_price=250, side="short")
    assert long_cost.total == pytest.approx(short_cost.total)


def test_round_trip_cost_at_price_rejects_bad_side():
    with pytest.raises(ValueError, match="long.*short"):
        round_trip_cost_at_price(qty=10, entry_price=100, exit_price=100, side="diagonal")


def test_slippage_inr_scales_with_notional_and_bps():
    assert slippage_inr(25_000, slippage_bps=5) == pytest.approx(12.5)
    assert slippage_inr(25_000, slippage_bps=10) == pytest.approx(25.0)
