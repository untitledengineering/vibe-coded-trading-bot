"""Tests for the streamer health monitor + reporting surface.

We don't spin up the real Upstox SDK here — we exercise the wrapping helpers
(`is_market_open`, `streamer_health`) and the watchdog logic in isolation.
"""

import time
from unittest.mock import MagicMock

import pytest

from src.api import streamer_manager


def _ts_for_ist(year, month, day, hour, minute) -> float:
    """Construct an epoch timestamp matching the given IST wall-clock moment."""
    import calendar
    utc_tuple = (year, month, day, hour, minute, 0, 0, 0, 0)
    epoch_in_ist_frame = calendar.timegm(utc_tuple)
    return epoch_in_ist_frame - streamer_manager.IST_OFFSET_SECONDS


# ----- is_market_open -----

def test_market_open_during_trading_hours():
    # Friday 2026-05-15 11:00 IST -> trading window
    assert streamer_manager.is_market_open(_ts_for_ist(2026, 5, 15, 11, 0)) is True


def test_market_closed_before_open():
    assert streamer_manager.is_market_open(_ts_for_ist(2026, 5, 15, 9, 14)) is False


def test_market_closed_at_close_time():
    # 15:30 IST is the close minute; first minute of "closed".
    assert streamer_manager.is_market_open(_ts_for_ist(2026, 5, 15, 15, 30)) is False


def test_market_closed_on_saturday():
    assert streamer_manager.is_market_open(_ts_for_ist(2026, 5, 16, 11, 0)) is False


def test_market_closed_on_sunday():
    assert streamer_manager.is_market_open(_ts_for_ist(2026, 5, 17, 11, 0)) is False


# ----- streamer_health -----

def test_health_when_no_streamer(monkeypatch):
    monkeypatch.setattr(streamer_manager, "streamer", None)
    monkeypatch.setattr(streamer_manager, "last_tick_at", 0.0)
    h = streamer_manager.streamer_health()
    assert h["streamer_running"] is False
    assert h["last_tick_seconds_ago"] is None
    assert h["stale"] is False


def test_health_when_streamer_running_with_recent_tick(monkeypatch):
    monkeypatch.setattr(streamer_manager, "streamer", MagicMock())
    monkeypatch.setattr(streamer_manager, "last_tick_at", time.time() - 2)
    h = streamer_manager.streamer_health()
    assert h["streamer_running"] is True
    assert 0 <= h["last_tick_seconds_ago"] <= 5
    assert h["stale"] is False


def test_health_marks_stale_during_market_hours_when_ticks_dry_up(monkeypatch):
    monkeypatch.setattr(streamer_manager, "streamer", MagicMock())
    monkeypatch.setattr(streamer_manager, "last_tick_at", time.time() - 300)
    monkeypatch.setattr(streamer_manager, "is_market_open", lambda *a, **kw: True)
    h = streamer_manager.streamer_health()
    assert h["stale"] is True


def test_health_not_stale_outside_market_hours(monkeypatch):
    monkeypatch.setattr(streamer_manager, "streamer", MagicMock())
    monkeypatch.setattr(streamer_manager, "last_tick_at", time.time() - 3600)
    monkeypatch.setattr(streamer_manager, "is_market_open", lambda *a, **kw: False)
    h = streamer_manager.streamer_health()
    assert h["stale"] is False


# ----- _fanout updates last_tick_at -----

def test_fanout_updates_last_tick_at(monkeypatch):
    monkeypatch.setattr(streamer_manager, "last_tick_at", 0.0)
    monkeypatch.setattr(streamer_manager.stream, "broadcast_tick", lambda _data: None)
    monkeypatch.setattr(streamer_manager, "bar_builder", None)
    streamer_manager._fanout({"feeds": {}})
    assert streamer_manager.last_tick_at > 0
    assert abs(streamer_manager.last_tick_at - time.time()) < 2
