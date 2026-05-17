"""Markdown report tests. We don't check pixel-perfect output — we check that
the load-bearing sections render with the right counts."""

from datetime import datetime

import pytest

from src.backtest.report import _diagnostic_observations, render_markdown
from src.backtest.runner import BacktestReport, EquityPoint
from src.bot.engine import EngineConfig
from src.bot.positions import Position


def _completed(side="long", pnl=10.0, reason="target"):
    p = Position(
        instrument_key="NSE_EQ|TEST",
        side=side, qty=10, entry_ts=60, entry_price=100.0,
        stop_loss_price=99.5, target_price=100.8,
    )
    p.exit_ts = 120
    p.exit_price = 100.0 + (pnl / 10)
    p.exit_reason = reason
    p.realised_pnl_inr = pnl
    return p


def _empty_report() -> BacktestReport:
    return BacktestReport(
        start_ts=1716000000, end_ts=1716086400,
        config=EngineConfig(), model_path="test",
    )


def test_render_handles_zero_trades():
    md = render_markdown(_empty_report())
    assert "Backtest report" in md
    assert "No trades were taken." in md
    assert "min_predicted_edge" in md  # the diagnostic hint references this knob


def test_render_emits_summary_for_nonzero_trades():
    r = _empty_report()
    r.completed_positions = [
        _completed(pnl=15.0, reason="target"),
        _completed(pnl=-8.0, reason="stop_loss"),
        _completed(pnl=12.0, reason="target"),
    ]
    r.equity_curve = [
        EquityPoint(minute_ts=60, cash_pnl=0, open_unrealised=0),
        EquityPoint(minute_ts=120, cash_pnl=19.0, open_unrealised=0),
    ]
    md = render_markdown(r)
    assert "Win rate" in md
    assert "trades: 3" in md
    assert "wins: 2" in md
    assert "losses: 1" in md


def test_diagnostic_observations_flags_coin_flip_win_rate():
    summary = {"trades": 50, "win_rate_pct": 50.0, "long_count": 25, "short_count": 25,
               "sl_exits": 25, "target_exits": 25}
    report = _empty_report()
    out = _diagnostic_observations(summary, daily=__import__("pandas").DataFrame(), report=report)
    joined = " ".join(out)
    assert "coin flip" in joined.lower()


def test_diagnostic_observations_flags_one_sided_book():
    summary = {"trades": 10, "win_rate_pct": 60.0, "long_count": 10, "short_count": 0,
               "sl_exits": 4, "target_exits": 6}
    report = _empty_report()
    out = _diagnostic_observations(summary, daily=__import__("pandas").DataFrame(), report=report)
    joined = " ".join(out)
    assert "one-sided" in joined.lower()
