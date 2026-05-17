"""Rule-based strategy logic tests."""

import numpy as np
import pandas as pd
import pytest

from src.bot.positions import Position
from src.features.technical import IST_OFFSET_SECONDS
from src.strategy.rule_based import (
    RuleBasedConfig,
    _clamp_long_sl,
    _clamp_short_sl,
    consecutive_losses,
    evaluate,
    is_entry_window,
    is_forced_exit,
)


def _bars_5m_with_indicator_state(closes, highs=None, lows=None):
    """Build a 5-min bars DataFrame for one symbol with controlled OHLC.
    minute_ts steps in 300s intervals starting at 09:15 IST day 20223."""
    n = len(closes)
    highs = highs if highs is not None else [c + 0.4 for c in closes]
    lows = lows if lows is not None else [c - 0.4 for c in closes]
    day_start_utc = 20223 * 86400 - IST_OFFSET_SECONDS
    minute_ts = [day_start_utc + 555 * 60 + 300 * i for i in range(n)]
    return pd.DataFrame({
        "minute_ts": minute_ts, "open": closes, "high": highs, "low": lows,
        "close": closes, "volume": [1000] * n,
    })


def _ts_for_ist(hour, minute, day_index=20223):
    return day_index * 86400 + (hour * 60 + minute) * 60 - IST_OFFSET_SECONDS


# ----- window helpers -----

def test_is_entry_window_inside_and_outside():
    cfg = RuleBasedConfig()
    assert is_entry_window(_ts_for_ist(10, 0), cfg) is True
    assert is_entry_window(_ts_for_ist(9, 15), cfg) is False  # before 09:20
    assert is_entry_window(_ts_for_ist(14, 35), cfg) is False  # after 14:30


def test_is_forced_exit_at_or_past_threshold():
    cfg = RuleBasedConfig()
    assert is_forced_exit(_ts_for_ist(15, 15), cfg) is True
    assert is_forced_exit(_ts_for_ist(15, 14), cfg) is False


# ----- SL clamping -----

def test_long_sl_uses_prev_low_when_within_cap():
    # entry 100, prev low 99.5 -> distance 0.5%, well within 1% cap -> use prev low
    sl = _clamp_long_sl(prev_low=99.5, entry_close=100.0, cap_pct=0.01)
    assert sl == pytest.approx(99.5)


def test_long_sl_caps_at_one_percent_when_prev_low_too_far():
    # entry 100, prev low 95 -> distance 5%, capped at 1% -> SL = 99
    sl = _clamp_long_sl(prev_low=95.0, entry_close=100.0, cap_pct=0.01)
    assert sl == pytest.approx(99.0)


def test_short_sl_uses_prev_high_when_within_cap():
    sl = _clamp_short_sl(prev_high=100.5, entry_close=100.0, cap_pct=0.01)
    assert sl == pytest.approx(100.5)


def test_short_sl_caps_at_one_percent_when_prev_high_too_far():
    sl = _clamp_short_sl(prev_high=105.0, entry_close=100.0, cap_pct=0.01)
    assert sl == pytest.approx(101.0)


# ----- evaluate(): main signal logic -----

def test_evaluate_emits_long_when_close_above_vwap_and_supertrend_green():
    # Rising sequence: close above VWAP, Supertrend should be green by the tail.
    closes = list(np.linspace(100, 115, 30))
    bars = _bars_5m_with_indicator_state(closes)
    cfg = RuleBasedConfig(max_positions=6)
    now = _ts_for_ist(10, 30)
    result = evaluate(
        bars_5m_by_symbol={"NSE_EQ|X": bars},
        open_positions=[], closed_today=[],
        now_ts=now, config=cfg,
    )
    assert len(result.intents) == 1
    intent = result.intents[0]
    assert intent.side == "long"
    assert intent.stop_loss_price is not None
    assert intent.target_price is not None
    # Target should be ~2x the SL distance from entry.
    entry_limit = closes[-1] * (1 + cfg.entry_buffer_pct)
    risk = entry_limit - intent.stop_loss_price
    expected_target = entry_limit + 2.0 * risk
    assert intent.target_price == pytest.approx(expected_target, rel=1e-6)


def test_evaluate_emits_short_when_close_below_vwap_and_supertrend_red():
    # Falling sequence: close should drop below VWAP, Supertrend turns red.
    closes = list(np.linspace(100, 85, 30))
    bars = _bars_5m_with_indicator_state(closes)
    cfg = RuleBasedConfig(max_positions=6)
    result = evaluate(
        bars_5m_by_symbol={"NSE_EQ|X": bars},
        open_positions=[], closed_today=[],
        now_ts=_ts_for_ist(10, 30), config=cfg,
    )
    assert len(result.intents) == 1
    assert result.intents[0].side == "short"


def test_evaluate_skips_symbols_already_held():
    closes = list(np.linspace(100, 115, 30))
    bars = _bars_5m_with_indicator_state(closes)
    held = Position(
        instrument_key="NSE_EQ|X", side="long", qty=10, entry_ts=_ts_for_ist(9, 30),
        entry_price=110.0, stop_loss_price=109.0, target_price=112.0,
    )
    result = evaluate(
        bars_5m_by_symbol={"NSE_EQ|X": bars},
        open_positions=[held], closed_today=[],
        now_ts=_ts_for_ist(10, 30), config=RuleBasedConfig(),
    )
    assert result.intents == []
    assert result.skipped.get("already_held") == 1


def test_evaluate_outside_entry_window_emits_nothing():
    bars = _bars_5m_with_indicator_state(list(np.linspace(100, 110, 30)))
    result = evaluate(
        bars_5m_by_symbol={"NSE_EQ|X": bars},
        open_positions=[], closed_today=[],
        now_ts=_ts_for_ist(8, 30), config=RuleBasedConfig(),  # before 09:20
    )
    assert result.intents == []
    assert result.skipped.get("outside_entry_window") == 1


def test_evaluate_respects_max_positions_cap():
    bars = _bars_5m_with_indicator_state(list(np.linspace(100, 115, 30)))
    cfg = RuleBasedConfig(max_positions=1)
    existing = Position(
        instrument_key="NSE_EQ|OTHER", side="long", qty=10, entry_ts=_ts_for_ist(9, 30),
        entry_price=200.0, stop_loss_price=199.0, target_price=202.0,
    )
    result = evaluate(
        bars_5m_by_symbol={"NSE_EQ|X": bars},
        open_positions=[existing], closed_today=[],
        now_ts=_ts_for_ist(10, 30), config=cfg,
    )
    assert result.intents == []
    assert result.skipped.get("max_concurrent_reached") == 1


def test_evaluate_drops_symbols_with_insufficient_history():
    short_bars = _bars_5m_with_indicator_state([100, 101, 102])  # < 10 bars
    result = evaluate(
        bars_5m_by_symbol={"NSE_EQ|X": short_bars},
        open_positions=[], closed_today=[],
        now_ts=_ts_for_ist(10, 30), config=RuleBasedConfig(),
    )
    assert result.intents == []
    assert result.skipped.get("insufficient_bars") == 1


# ----- consecutive_losses -----

def test_consecutive_losses_counts_trailing_losers():
    p1 = Position("X", "long", 10, 1, 100, 99, 102,
                  exit_ts=2, exit_price=99, exit_reason="stop_loss", realised_pnl_inr=-5)
    p2 = Position("X", "long", 10, 3, 100, 99, 102,
                  exit_ts=4, exit_price=101, exit_reason="target", realised_pnl_inr=10)
    p3 = Position("X", "long", 10, 5, 100, 99, 102,
                  exit_ts=6, exit_price=99, exit_reason="stop_loss", realised_pnl_inr=-5)
    p4 = Position("X", "long", 10, 7, 100, 99, 102,
                  exit_ts=8, exit_price=99, exit_reason="stop_loss", realised_pnl_inr=-5)
    # Trailing sequence (sorted by exit_ts): p1 LOSS, p2 WIN, p3 LOSS, p4 LOSS.
    # consec losses at tail = 2 (p3, p4).
    assert consecutive_losses([p1, p2, p3, p4]) == 2


def test_consecutive_losses_zero_when_last_is_win():
    p1 = Position("X", "long", 10, 1, 100, 99, 102,
                  exit_ts=2, exit_price=99, exit_reason="stop_loss", realised_pnl_inr=-5)
    p2 = Position("X", "long", 10, 3, 100, 99, 102,
                  exit_ts=4, exit_price=101, exit_reason="target", realised_pnl_inr=10)
    assert consecutive_losses([p1, p2]) == 0


# ----- Position.maybe_trail_to_breakeven -----

def test_position_trails_to_breakeven_on_1R_long():
    p = Position("X", "long", 10, 0, 100.0, 99.0, 102.0)  # risk=1, target=+2R
    # At price 101, profit per share = 1 = risk -> trail trigger
    moved = p.maybe_trail_to_breakeven(current_price=101.0)
    assert moved is True
    assert p.stop_loss_price == pytest.approx(100.0)
    assert p.breakeven_locked is True


def test_position_trails_to_breakeven_on_1R_short():
    p = Position("X", "short", 10, 0, 100.0, 101.0, 98.0)  # risk=1
    moved = p.maybe_trail_to_breakeven(current_price=99.0)
    assert moved is True
    assert p.stop_loss_price == pytest.approx(100.0)


def test_position_does_not_trail_below_1R():
    p = Position("X", "long", 10, 0, 100.0, 99.0, 102.0)
    moved = p.maybe_trail_to_breakeven(current_price=100.5)
    assert moved is False
    assert p.stop_loss_price == 99.0


def test_position_breakeven_idempotent():
    p = Position("X", "long", 10, 0, 100.0, 99.0, 102.0)
    p.maybe_trail_to_breakeven(101.0)
    second = p.maybe_trail_to_breakeven(102.0)
    assert second is False  # already locked
