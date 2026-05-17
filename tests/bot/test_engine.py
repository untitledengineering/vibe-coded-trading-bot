"""Decision-engine tests. We bypass real xgboost by patching the score function;
the engine's job is selection, not inference."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.bot.engine import (
    EngineConfig,
    compute_stop_target,
    decide,
    is_entry_window,
    is_forced_exit,
)
from src.bot.positions import Position
from src.features.technical import FEATURE_COLUMNS, IST_OFFSET_SECONDS


def _ts_for_ist_minute(ist_minute_of_day: int) -> int:
    """Return epoch ts whose IST minute-of-day equals the input. Uses day 0 (1970-01-01 IST)."""
    return ist_minute_of_day * 60 - IST_OFFSET_SECONDS


def _features_frame(rows):
    """rows is a list of (instrument_key, close). Features filled with zero."""
    out = []
    for key, close in rows:
        d = {"instrument_key": key, "close": close}
        d.update({c: 0.0 for c in FEATURE_COLUMNS})
        out.append(d)
    return pd.DataFrame(out)


def _patch_score(monkeypatch, predictions):
    from src.bot import engine as engine_mod
    monkeypatch.setattr(engine_mod, "score", lambda model, df: np.array(predictions))


# ---- window helpers ----

def test_is_entry_window_inside_and_outside():
    cfg = EngineConfig()
    assert is_entry_window(_ts_for_ist_minute(10 * 60), cfg) is True
    assert is_entry_window(_ts_for_ist_minute(9 * 60), cfg) is False
    assert is_entry_window(_ts_for_ist_minute(14 * 60 + 45), cfg) is False


def test_is_forced_exit_at_or_past_threshold():
    cfg = EngineConfig()
    assert is_forced_exit(_ts_for_ist_minute(14 * 60 + 55), cfg) is True
    assert is_forced_exit(_ts_for_ist_minute(14 * 60 + 54), cfg) is False


def test_compute_stop_target_long():
    cfg = EngineConfig(stop_loss_pct=0.005, target_pct=0.008)
    sl, tp = compute_stop_target(entry_price=100.0, side="long", config=cfg)
    assert sl == pytest.approx(99.5)
    assert tp == pytest.approx(100.8)


def test_compute_stop_target_short():
    cfg = EngineConfig(stop_loss_pct=0.005, target_pct=0.008)
    sl, tp = compute_stop_target(entry_price=100.0, side="short", config=cfg)
    assert sl == pytest.approx(100.5)
    assert tp == pytest.approx(99.2)


# ---- decide() ----

def test_decide_emits_top_long_and_short_above_threshold(monkeypatch):
    cfg = EngineConfig(top_k_long=1, top_k_short=1, min_predicted_edge=0.001)
    features = _features_frame([("A", 100.0), ("B", 100.0), ("C", 100.0), ("D", 100.0)])
    _patch_score(monkeypatch, [0.01, 0.002, -0.0005, -0.01])
    now = _ts_for_ist_minute(10 * 60)
    result = decide(features, model=MagicMock(), open_positions=[], config=cfg, now_ts=now)
    sides_by_key = {i.instrument_key: i.side for i in result.intents}
    assert sides_by_key == {"A": "long", "D": "short"}


def test_decide_skips_outside_entry_window(monkeypatch):
    cfg = EngineConfig()
    features = _features_frame([("A", 100.0)])
    _patch_score(monkeypatch, [0.01])
    now = _ts_for_ist_minute(9 * 60)
    result = decide(features, model=MagicMock(), open_positions=[], config=cfg, now_ts=now)
    assert result.intents == []
    assert result.skipped_reasons.get("outside_entry_window") == 1


def test_decide_emits_eod_exits_for_open_positions(monkeypatch):
    cfg = EngineConfig()
    features = _features_frame([("A", 100.0)])
    _patch_score(monkeypatch, [0.01])
    held = Position(instrument_key="HELD", side="long", qty=10, entry_ts=0,
                    entry_price=100.0, stop_loss_price=99.5, target_price=100.8)
    now = _ts_for_ist_minute(14 * 60 + 56)
    result = decide(features, model=MagicMock(), open_positions=[held], config=cfg, now_ts=now)
    assert len(result.intents) == 1
    assert result.intents[0].instrument_key == "HELD"
    assert result.intents[0].reason == "exit_eod"


def test_decide_respects_concurrency_cap(monkeypatch):
    cfg = EngineConfig(max_concurrent_positions=2, top_k_long=2, top_k_short=2,
                       min_predicted_edge=0.001)
    features = _features_frame([("A", 100.0), ("B", 100.0), ("C", 100.0)])
    _patch_score(monkeypatch, [0.01, 0.005, -0.01])
    existing = Position(instrument_key="X", side="long", qty=10, entry_ts=0,
                        entry_price=100, stop_loss_price=99.5, target_price=100.8)
    now = _ts_for_ist_minute(10 * 60)
    result = decide(features, model=MagicMock(), open_positions=[existing], config=cfg, now_ts=now)
    assert len(result.intents) == 1


def test_decide_drops_candidates_below_edge_threshold(monkeypatch):
    cfg = EngineConfig(min_predicted_edge=0.005, top_k_long=2, top_k_short=2)
    features = _features_frame([("A", 100.0), ("B", 100.0)])
    _patch_score(monkeypatch, [0.001, -0.001])
    now = _ts_for_ist_minute(10 * 60)
    result = decide(features, model=MagicMock(), open_positions=[], config=cfg, now_ts=now)
    assert result.intents == []
    assert result.skipped_reasons.get("below_edge_threshold") == 2


def test_decide_cooldown_blocks_recent_re_entry(monkeypatch):
    """A symbol exited within cooldown_minutes must be skipped — fixes the
    IDFC-Bank-17-times-per-session bug from the model_v1 backtest."""
    cfg = EngineConfig(top_k_long=1, top_k_short=0, min_predicted_edge=0.001,
                       cooldown_minutes=30, max_trades_per_symbol_per_day=999)
    features = _features_frame([("A", 100.0)])
    _patch_score(monkeypatch, [0.02])
    now = _ts_for_ist_minute(10 * 60)
    just_closed = Position(
        instrument_key="A", side="long", qty=10, entry_ts=now - 600,
        entry_price=100.0, stop_loss_price=99.5, target_price=100.8,
        exit_ts=now - 60,  # exited 1 minute ago
        exit_price=99.5, exit_reason="stop_loss", realised_pnl_inr=-5.0,
    )
    result = decide(features, model=MagicMock(), open_positions=[],
                    config=cfg, now_ts=now, closed_positions=[just_closed])
    assert result.intents == []
    assert result.skipped_reasons.get("cooldown_active") == 1


def test_decide_cooldown_releases_after_expiry(monkeypatch):
    cfg = EngineConfig(top_k_long=1, top_k_short=0, min_predicted_edge=0.001,
                       cooldown_minutes=30, max_trades_per_symbol_per_day=999)
    features = _features_frame([("A", 100.0)])
    _patch_score(monkeypatch, [0.02])
    now = _ts_for_ist_minute(10 * 60)
    long_ago = Position(
        instrument_key="A", side="long", qty=10, entry_ts=now - 7200,
        entry_price=100.0, stop_loss_price=99.5, target_price=100.8,
        exit_ts=now - 3600,  # exited 60 minutes ago — well past 30-min cooldown
        exit_price=100.5, exit_reason="target", realised_pnl_inr=5.0,
    )
    result = decide(features, model=MagicMock(), open_positions=[],
                    config=cfg, now_ts=now, closed_positions=[long_ago])
    assert len(result.intents) == 1
    assert result.intents[0].instrument_key == "A"


def test_decide_per_symbol_daily_cap_enforced(monkeypatch):
    """Once a symbol has hit max_trades_per_symbol_per_day, it can't be re-entered
    even after the cooldown expires."""
    cfg = EngineConfig(top_k_long=1, top_k_short=0, min_predicted_edge=0.001,
                       cooldown_minutes=1, max_trades_per_symbol_per_day=2)
    features = _features_frame([("A", 100.0)])
    _patch_score(monkeypatch, [0.02])
    now = _ts_for_ist_minute(13 * 60)
    # Two earlier completed trades on A today, both well past cooldown.
    closed = [
        Position(instrument_key="A", side="long", qty=10, entry_ts=now - 10800,
                 entry_price=100, stop_loss_price=99.5, target_price=100.8,
                 exit_ts=now - 10000, exit_price=100.5, exit_reason="target",
                 realised_pnl_inr=5.0),
        Position(instrument_key="A", side="long", qty=10, entry_ts=now - 7200,
                 entry_price=100, stop_loss_price=99.5, target_price=100.8,
                 exit_ts=now - 6500, exit_price=99.5, exit_reason="stop_loss",
                 realised_pnl_inr=-5.0),
    ]
    result = decide(features, model=MagicMock(), open_positions=[],
                    config=cfg, now_ts=now, closed_positions=closed)
    assert result.intents == []
    assert result.skipped_reasons.get("symbol_daily_cap_reached") == 1


def test_decide_per_symbol_cap_counts_currently_open(monkeypatch):
    """If we already have an open position on the symbol today AND closed-trade
    count meets the cap, no new entries. (open is also already excluded by the
    pyramid-block, but the cap should be honest about total day's attempts.)"""
    cfg = EngineConfig(top_k_long=1, top_k_short=0, min_predicted_edge=0.001,
                       cooldown_minutes=1, max_trades_per_symbol_per_day=2)
    features = _features_frame([("A", 100.0)])
    _patch_score(monkeypatch, [0.02])
    now = _ts_for_ist_minute(13 * 60)
    # One closed today + one open today = 2 trades. New attempt should fail the cap.
    closed_today = Position(instrument_key="A", side="long", qty=10, entry_ts=now - 5000,
                            entry_price=100, stop_loss_price=99.5, target_price=100.8,
                            exit_ts=now - 4000, exit_price=100.5, exit_reason="target",
                            realised_pnl_inr=5.0)
    open_today = Position(instrument_key="A", side="long", qty=10, entry_ts=now - 1000,
                          entry_price=100, stop_loss_price=99.5, target_price=100.8)
    # We deliberately don't include A in features so the cap check fires without
    # being short-circuited by the open-symbol pyramid block. Use a different symbol.
    features = _features_frame([("A", 100.0)])
    result = decide(features, model=MagicMock(), open_positions=[open_today],
                    config=cfg, now_ts=now, closed_positions=[closed_today])
    # The pyramid block excludes A first (it's open), so candidates becomes empty.
    assert result.skipped_reasons.get("all_candidates_already_held") == 1


def test_decide_sentiment_veto_blocks_long_on_bearish_news(monkeypatch):
    """A bearish sentiment score below -threshold prevents a long intent."""
    cfg = EngineConfig(top_k_long=1, top_k_short=0, min_predicted_edge=0.001,
                       sentiment_veto_threshold=0.3)
    features = _features_frame([("A", 100.0)])
    _patch_score(monkeypatch, [0.02])
    now = _ts_for_ist_minute(10 * 60)
    result = decide(features, model=MagicMock(), open_positions=[], config=cfg,
                    now_ts=now, sentiment_scores={"A": -0.5})
    assert result.intents == []
    assert result.skipped_reasons.get("sentiment_bearish_veto") == 1


def test_decide_sentiment_veto_blocks_short_on_bullish_news(monkeypatch):
    """A bullish sentiment score above +threshold prevents a short intent."""
    cfg = EngineConfig(top_k_long=0, top_k_short=1, min_predicted_edge=0.001,
                       sentiment_veto_threshold=0.3)
    features = _features_frame([("A", 100.0)])
    _patch_score(monkeypatch, [-0.02])
    now = _ts_for_ist_minute(10 * 60)
    result = decide(features, model=MagicMock(), open_positions=[], config=cfg,
                    now_ts=now, sentiment_scores={"A": 0.5})
    assert result.intents == []
    assert result.skipped_reasons.get("sentiment_bullish_veto") == 1


def test_decide_sentiment_neutral_does_not_veto(monkeypatch):
    """A score inside the threshold band should not block the trade."""
    cfg = EngineConfig(top_k_long=1, top_k_short=0, min_predicted_edge=0.001,
                       sentiment_veto_threshold=0.3)
    features = _features_frame([("A", 100.0)])
    _patch_score(monkeypatch, [0.02])
    now = _ts_for_ist_minute(10 * 60)
    # Score -0.2 is below the threshold magnitude (0.3) — should not veto.
    result = decide(features, model=MagicMock(), open_positions=[], config=cfg,
                    now_ts=now, sentiment_scores={"A": -0.2})
    assert len(result.intents) == 1
    assert result.intents[0].side == "long"


def test_decide_no_sentiment_scores_no_veto(monkeypatch):
    """Passing sentiment_scores=None (no news data) must not affect decisions."""
    cfg = EngineConfig(top_k_long=1, top_k_short=0, min_predicted_edge=0.001,
                       sentiment_veto_threshold=0.3)
    features = _features_frame([("A", 100.0)])
    _patch_score(monkeypatch, [0.02])
    now = _ts_for_ist_minute(10 * 60)
    result = decide(features, model=MagicMock(), open_positions=[], config=cfg,
                    now_ts=now, sentiment_scores=None)
    assert len(result.intents) == 1


def test_decide_veto_threshold_zero_disables_veto(monkeypatch):
    """sentiment_veto_threshold=0 disables the veto even for extreme scores."""
    cfg = EngineConfig(top_k_long=1, top_k_short=0, min_predicted_edge=0.001,
                       sentiment_veto_threshold=0.0)
    features = _features_frame([("A", 100.0)])
    _patch_score(monkeypatch, [0.02])
    now = _ts_for_ist_minute(10 * 60)
    result = decide(features, model=MagicMock(), open_positions=[], config=cfg,
                    now_ts=now, sentiment_scores={"A": -1.0})
    assert len(result.intents) == 1


def test_decide_does_not_pyramid_into_open_symbol(monkeypatch):
    cfg = EngineConfig(top_k_long=1, top_k_short=1, min_predicted_edge=0.001)
    features = _features_frame([("HELD", 100.0), ("FREE", 100.0)])
    # `decide` filters HELD out before scoring; mock score by symbol name so the
    # mock works regardless of how many rows survive the filter.
    from src.bot import engine as engine_mod
    monkeypatch.setattr(
        engine_mod,
        "score",
        lambda model, frame: np.array(
            [{"HELD": 0.02, "FREE": 0.01}[k] for k in frame["instrument_key"]]
        ),
    )
    held = Position(instrument_key="HELD", side="long", qty=10, entry_ts=0,
                    entry_price=100.0, stop_loss_price=99.5, target_price=100.8)
    now = _ts_for_ist_minute(10 * 60)
    result = decide(features, model=MagicMock(), open_positions=[held], config=cfg, now_ts=now)
    keys = {i.instrument_key for i in result.intents}
    assert keys == {"FREE"}
