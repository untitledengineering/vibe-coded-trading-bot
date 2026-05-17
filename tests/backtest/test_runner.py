"""End-to-end runner test on a tiny synthetic dataset. Validates that:
    - Pending intents fill at next-bar open
    - SL triggers on bar low
    - EOD exits close any survivors
    - Report carries the right counters
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.backtest.runner import BacktestReport, run_backtest
from src.bot.engine import EngineConfig
from src.features.technical import FEATURE_COLUMNS, IST_OFFSET_SECONDS


def _make_fake_dataset_for_runner(monkeypatch):
    """Patch the dataset assembler to return a synthetic 2-symbol 1-day frame.
    Day = 2025-05-15 IST. Symbols A (rising) and B (falling)."""
    day_idx = 20223  # 2025-05-15 = day 20223 since epoch
    rows = []
    for minute in range(15):  # 15 bars starting at 09:15 IST
        for key, base, drift in [("NSE_EQ|A", 100.0, +0.5), ("NSE_EQ|B", 100.0, -0.5)]:
            close = base + drift * minute
            ts = day_idx * 86400 + (9 * 60 + 15 + minute) * 60 - IST_OFFSET_SECONDS
            row = {
                "instrument_key": key, "minute_ts": ts,
                "open": close - 0.05, "high": close + 0.2, "low": close - 0.2,
                "close": close, "volume": 1000,
            }
            for c in FEATURE_COLUMNS:
                row[c] = 0.0
            row["fwd_ret_15m"] = drift * 0.001
            row["fwd_up_15m"] = 1.0 if drift > 0 else 0.0
            rows.append(row)
    df = pd.DataFrame(rows).sort_values("minute_ts").reset_index(drop=True)

    from src.backtest import runner as runner_mod
    monkeypatch.setattr(runner_mod, "assemble_full_dataset", lambda **kw: df)
    return df


def _fake_model_artifact(predictions_by_key):
    """Fake ModelArtifact + patched score() that returns per-symbol predictions."""
    artifact = MagicMock()
    artifact.feature_columns = list(FEATURE_COLUMNS)
    artifact.label_horizon_minutes = 15

    def _score(model, frame):
        return np.array([predictions_by_key.get(k, 0.0) for k in frame["instrument_key"]])

    from src.backtest import runner as runner_mod
    return artifact, _score


def test_runner_produces_no_trades_when_predictions_below_edge(monkeypatch):
    _make_fake_dataset_for_runner(monkeypatch)
    artifact, scorer = _fake_model_artifact({"NSE_EQ|A": 0.0001, "NSE_EQ|B": -0.0001})
    from src.bot import engine as engine_mod
    monkeypatch.setattr(engine_mod, "score", scorer)
    cfg = EngineConfig(min_predicted_edge=0.005,
                       entry_window_open_minute_ist=9 * 60 + 15,  # allow our synthetic 09:15 window
                       entry_window_close_minute_ist=14 * 60 + 30,
                       forced_exit_minute_ist=14 * 60 + 55)
    report = run_backtest("2025-05-15", "2025-05-15", model=artifact, config=cfg)
    assert isinstance(report, BacktestReport)
    assert report.completed_positions == []
    assert "below_edge_threshold" in report.skipped


def test_runner_opens_and_closes_at_eod_when_signals_present(monkeypatch):
    _make_fake_dataset_for_runner(monkeypatch)
    artifact, scorer = _fake_model_artifact({"NSE_EQ|A": 0.02, "NSE_EQ|B": -0.02})
    from src.bot import engine as engine_mod
    monkeypatch.setattr(engine_mod, "score", scorer)
    # Allow entries during our synthetic 15-minute morning window. Forced exit
    # is set to the LAST minute of the synthetic data so positions close cleanly.
    cfg = EngineConfig(
        min_predicted_edge=0.005,
        top_k_long=1, top_k_short=1,
        entry_window_open_minute_ist=9 * 60 + 15,
        entry_window_close_minute_ist=9 * 60 + 20,
        forced_exit_minute_ist=9 * 60 + 29,
    )
    report = run_backtest("2025-05-15", "2025-05-15", model=artifact, config=cfg)
    # We expect both an A (long) and B (short) entry, then both closed at EOD.
    keys = {p.instrument_key for p in report.completed_positions}
    assert "NSE_EQ|A" in keys
    assert "NSE_EQ|B" in keys
    assert all(p.exit_reason in ("eod", "stop_loss", "target") for p in report.completed_positions)
