"""SpreadPosition math — critical to get right because the live engine will
compute P&L every tick on these formulas."""

import pytest

from src.options.positions import SpreadPosition, estimate_spread_margin_inr


def _bull_put(short_strike=23500, long_strike=23400, short_price=80, long_price=40,
              qty_lots=1, lot_size=50):
    """Bull put: sell PUT at 23500 (collect 80), buy PUT at 23400 (pay 40).
    Net credit = 40. Spread width = 100. Max loss per lot = 60."""
    return SpreadPosition(
        underlying="NSE_INDEX|Nifty 50", spread_type="bull_put_spread",
        expiry_date="2026-05-22", entry_ts=1_000_000,
        short_leg_key="NSE_FO|SHORT", short_strike=short_strike, short_entry_price=short_price,
        long_leg_key="NSE_FO|LONG", long_strike=long_strike, long_entry_price=long_price,
        qty_lots=qty_lots, lot_size=lot_size,
    )


def test_credit_received_per_lot():
    s = _bull_put(short_price=80, long_price=40)
    assert s.credit_received_per_lot == pytest.approx(40.0)


def test_spread_width_is_absolute():
    s = _bull_put(short_strike=23500, long_strike=23400)
    assert s.spread_width == 100.0


def test_max_loss_per_lot_is_width_minus_credit():
    s = _bull_put(short_strike=23500, long_strike=23400, short_price=80, long_price=40)
    # width 100, credit 40 -> max loss per lot = 60 points
    assert s.max_loss_per_lot == pytest.approx(60.0)


def test_max_loss_inr_scales_with_lots_and_lot_size():
    s = _bull_put(qty_lots=2, lot_size=50)
    # max_loss_per_lot=60, qty=2, lot_size=50 -> 60 * 2 * 50 = ₹6,000
    assert s.max_loss_inr == pytest.approx(6000.0)


def test_max_profit_inr_equals_credit_collected():
    s = _bull_put(qty_lots=1, lot_size=50)
    # credit 40 * 1 * 50 = ₹2,000
    assert s.max_profit_inr == pytest.approx(2000.0)


def test_unrealised_pnl_at_entry_is_zero():
    s = _bull_put(short_price=80, long_price=40)
    assert s.unrealised_pnl_inr(short_mark=80, long_mark=40) == 0.0


def test_unrealised_pnl_when_premia_decay_favourably():
    # We sold short @ 80 — if short premium drops to 30, we'd buy back for 30
    # = profit of 50 per lot. Long @ 40 -> say long now 10 means we close it for 10,
    # locking in a loss of 30. Net per lot = +50 − 30 = +20.
    s = _bull_put(short_price=80, long_price=40, qty_lots=1, lot_size=50)
    pnl = s.unrealised_pnl_inr(short_mark=30, long_mark=10)
    assert pnl == pytest.approx(20 * 50)


def test_unrealised_pnl_when_market_runs_against_us():
    # Spot crashes, short put gains value (we lose). Short 80 -> 150 = -70/lot.
    # Long 40 -> 60 = +20/lot. Net = -50/lot.
    s = _bull_put(qty_lots=1, lot_size=50)
    pnl = s.unrealised_pnl_inr(short_mark=150, long_mark=60)
    assert pnl == pytest.approx(-50 * 50)


def test_close_writes_realised_pnl():
    s = _bull_put(short_price=80, long_price=40, qty_lots=1, lot_size=50)
    s.close(exit_ts=2_000_000, short_exit_price=30, long_exit_price=10,
            reason="target", costs_inr=100.0)
    # gross = +20/lot × 50 = 1000, minus 100 costs = 900
    assert s.realised_pnl_inr == pytest.approx(900.0)
    assert s.exit_reason == "target"
    assert s.is_open is False


def test_estimate_spread_margin_inr_matches_max_loss():
    s = _bull_put(qty_lots=2, lot_size=50)
    assert estimate_spread_margin_inr(s) == s.max_loss_inr
