"""VWAP + Supertrend tests on synthetic data with known answers."""

import numpy as np
import pandas as pd
import pytest

from src.features.technical import IST_OFFSET_SECONDS
from src.strategy.indicators import session_vwap, supertrend


def _session_bars(day_index: int, closes, highs=None, lows=None, vols=None):
    n = len(closes)
    highs = highs if highs is not None else [c + 0.5 for c in closes]
    lows = lows if lows is not None else [c - 0.5 for c in closes]
    vols = vols if vols is not None else [100] * n
    day_start_utc = day_index * 86400 - IST_OFFSET_SECONDS
    minute_ts = [day_start_utc + 555 * 60 + 60 * i for i in range(n)]
    return pd.DataFrame({
        "minute_ts": minute_ts, "open": closes, "high": highs, "low": lows,
        "close": closes, "volume": vols,
    })


# ----- VWAP -----

def test_vwap_equals_price_when_price_flat():
    bars = _session_bars(20223, [100.0] * 20)
    vwap = session_vwap(bars)
    # tp = (h+l+c)/3 = 100 when h=l=c=100. With h=c+0.5 l=c-0.5, tp = c = 100.
    assert np.allclose(vwap.dropna().values, 100.0)


def test_vwap_rises_with_late_high_volume():
    closes = [100.0] * 10 + [110.0] * 10  # price jumps in second half
    vols = [100] * 10 + [500] * 10        # plus volume skew
    bars = _session_bars(20223, closes, vols=vols)
    vwap = session_vwap(bars)
    # First bar VWAP ~100; last bar VWAP weighted heavily toward 110.
    assert vwap.iloc[0] == pytest.approx(100.0)
    assert 105 < vwap.iloc[-1] < 110


# ----- Supertrend -----

def test_supertrend_starts_uptrend_on_steadily_rising_close():
    closes = list(np.linspace(100, 120, 40))
    bars = _session_bars(20223, closes)
    _st_line, direction = supertrend(bars["high"], bars["low"], bars["close"], period=7)
    # After warmup, direction should be solidly +1 throughout the steady rise.
    valid = direction.dropna() if direction.dtype != np.int8 else direction
    tail = direction.iloc[10:]
    assert (tail == 1).all()


def test_supertrend_flips_to_downtrend_after_sharp_reversal():
    up = list(np.linspace(100, 120, 40))
    down = list(np.linspace(119, 90, 40))
    closes = up + down
    bars = _session_bars(20223, closes)
    _st_line, direction = supertrend(bars["high"], bars["low"], bars["close"], period=7)
    # First half ends in uptrend...
    assert direction.iloc[35] == 1
    # ...and we eventually flip to -1 during the down leg.
    assert (direction.iloc[40:] == -1).any()


def test_supertrend_handles_short_input_gracefully():
    closes = list(np.linspace(100, 105, 5))  # shorter than period=7
    bars = _session_bars(20223, closes)
    st_line, direction = supertrend(bars["high"], bars["low"], bars["close"], period=7)
    # ATR is NaN, so supertrend stays at its initial zeros — no exception.
    assert len(st_line) == 5
    assert len(direction) == 5
