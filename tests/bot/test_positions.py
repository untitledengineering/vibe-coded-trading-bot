"""Position-tracker tests. Pure data + small helpers — we test every branch."""

import pytest

from src.bot.positions import Position


def _long_pos(entry=100.0, qty=10, sl=99.5, tp=100.8):
    return Position(
        instrument_key="X", side="long", qty=qty, entry_ts=0, entry_price=entry,
        stop_loss_price=sl, target_price=tp,
    )


def _short_pos(entry=100.0, qty=10, sl=100.5, tp=99.2):
    return Position(
        instrument_key="X", side="short", qty=qty, entry_ts=0, entry_price=entry,
        stop_loss_price=sl, target_price=tp,
    )


def test_unrealised_pnl_long():
    p = _long_pos()
    assert p.unrealised_pnl_inr(102.0) == pytest.approx(10 * (102 - 100))
    assert p.unrealised_pnl_inr(98.0) == pytest.approx(10 * (98 - 100))


def test_unrealised_pnl_short_flips_sign():
    p = _short_pos()
    assert p.unrealised_pnl_inr(98.0) == pytest.approx(10 * (100 - 98))
    assert p.unrealised_pnl_inr(102.0) == pytest.approx(10 * (100 - 102))


def test_should_exit_long_at_sl_when_low_breaches():
    p = _long_pos(sl=99.5, tp=100.8)
    assert p.should_exit_at(high=100.0, low=99.4) == "stop_loss"


def test_should_exit_long_at_target_when_high_breaches():
    p = _long_pos(sl=99.5, tp=100.8)
    assert p.should_exit_at(high=100.9, low=99.9) == "target"


def test_should_exit_long_returns_none_when_neither_hit():
    p = _long_pos(sl=99.5, tp=100.8)
    assert p.should_exit_at(high=100.7, low=99.6) is None


def test_should_exit_short_at_sl_when_high_breaches():
    p = _short_pos(sl=100.5, tp=99.2)
    assert p.should_exit_at(high=100.6, low=100.0) == "stop_loss"


def test_should_exit_short_at_target_when_low_breaches():
    p = _short_pos(sl=100.5, tp=99.2)
    assert p.should_exit_at(high=100.2, low=99.1) == "target"


def test_sl_takes_priority_over_tp_when_both_hit_same_bar():
    """If a single bar's range spans both SL and TP, we close at SL — conservative."""
    p = _long_pos(sl=99.5, tp=100.8)
    assert p.should_exit_at(high=101.0, low=99.0) == "stop_loss"


def test_close_records_pnl_after_costs():
    p = _long_pos(entry=100.0, qty=10)
    p.close(exit_ts=60, exit_price=102.0, reason="target", costs_inr=5.0)
    # Gross = 10 × (102-100) = 20. Net = 20 - 5 = 15.
    assert p.realised_pnl_inr == pytest.approx(15.0)
    assert p.exit_reason == "target"
    assert p.is_open is False
