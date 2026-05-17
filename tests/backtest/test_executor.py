"""Backtest-executor tests. Fills + slippage + cost wiring."""

import pytest

from src.backtest.executor import BacktestExecutor
from src.bot.engine import EngineConfig, OrderIntent


def test_open_position_applies_slippage_against_long():
    ex = BacktestExecutor(config=EngineConfig(), slippage_bps=5.0)
    intent = OrderIntent(instrument_key="X", side="long", qty=10,
                         reason="long_top_pick", predicted_return=0.01)
    pos = ex.open_position(intent, fill_ts=60, bar_open_price=100.0)
    # 5 bps above 100 = 100.05.
    assert pos.entry_price == pytest.approx(100.05)
    assert pos.qty == 10
    # SL/TP derived from EngineConfig defaults: -0.5% / +0.8% of entry.
    assert pos.stop_loss_price == pytest.approx(100.05 * (1 - 0.005))
    assert pos.target_price == pytest.approx(100.05 * (1 + 0.008))


def test_open_position_applies_slippage_against_short():
    ex = BacktestExecutor(config=EngineConfig(), slippage_bps=5.0)
    intent = OrderIntent(instrument_key="X", side="short", qty=10,
                         reason="short_bottom_pick", predicted_return=-0.01)
    pos = ex.open_position(intent, fill_ts=60, bar_open_price=100.0)
    # 5 bps below 100 = 99.95.
    assert pos.entry_price == pytest.approx(99.95)


def test_close_position_records_pnl_minus_costs():
    ex = BacktestExecutor(config=EngineConfig(), slippage_bps=5.0)
    intent = OrderIntent(instrument_key="X", side="long", qty=100,
                         reason="long_top_pick", predicted_return=0.01)
    pos = ex.open_position(intent, fill_ts=60, bar_open_price=100.0)
    # Close at higher price 102. Slippage applies against us (long sells lower).
    ex.close_position(pos, exit_ts=120, exit_price=102.0, reason="target")
    assert pos.exit_reason == "target"
    assert pos.realised_pnl_inr is not None
    # Gross gain should be roughly 100 × (102 - 100.05) before costs ≈ ₹195 minus
    # costs (~₹30-50 for this notional) and slippage drag at exit.
    assert 100 < pos.realised_pnl_inr < 200
