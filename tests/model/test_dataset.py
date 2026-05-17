"""Tests for dataset assembly: end-to-end on synthetic bars + temporal split."""

import numpy as np
import pandas as pd
import pytest

from src.features.technical import FEATURE_COLUMNS, IST_OFFSET_SECONDS
from src.model.dataset import (
    DatasetSpec,
    build_dataset_for_symbol,
    time_split,
    x_y_arrays,
)


def _make_session(day_index: int, n_bars: int, base_close: float = 100.0):
    closes = base_close + np.arange(n_bars, dtype=float) * 0.05
    day_start_utc = day_index * 86400 - IST_OFFSET_SECONDS
    minute_ts = [day_start_utc + 555 * 60 + 60 * i for i in range(n_bars)]
    return pd.DataFrame(
        {
            "minute_ts": minute_ts,
            "open": closes,
            "high": closes + 0.1,
            "low": closes - 0.1,
            "close": closes,
            "volume": np.random.default_rng(seed=day_index).integers(100, 1000, size=n_bars),
        }
    )


def test_build_dataset_for_symbol_drops_warmup_and_boundary_nans():
    # 60 sessions of 100 bars each so vol_z_20d also fills in.
    bars = pd.concat([_make_session(20000 + i, 100) for i in range(60)], ignore_index=True)
    spec = DatasetSpec()
    out = build_dataset_for_symbol(bars, "NSE_EQ|TEST", spec=spec)
    assert not out.empty
    # Output must have no NaN in features or in the regression label.
    for col in FEATURE_COLUMNS:
        assert out[col].notna().all(), f"NaN leaked through in column {col}"
    assert out[spec.label_regression_column].notna().all()
    # instrument_key was injected.
    assert (out["instrument_key"] == "NSE_EQ|TEST").all()


def test_build_dataset_for_symbol_returns_empty_for_empty_input():
    out = build_dataset_for_symbol(pd.DataFrame(), "NSE_EQ|X")
    assert out.empty


def test_time_split_strictly_temporal():
    bars = _make_session(20000, 100)
    spec = DatasetSpec()
    bars["fwd_ret_15m"] = 0.0
    bars["fwd_up_15m"] = 0.0
    for c in FEATURE_COLUMNS:
        bars[c] = 0.0
    bars["instrument_key"] = "X"
    split_ts = bars["minute_ts"].iloc[50]
    train, test = time_split(bars, train_end_ts=split_ts)
    assert (train["minute_ts"] <= split_ts).all()
    assert (test["minute_ts"] > split_ts).all()
    # Together they cover everything.
    assert len(train) + len(test) == len(bars)


def test_x_y_arrays_pulls_correct_columns():
    bars = pd.concat([_make_session(20000 + i, 100) for i in range(60)], ignore_index=True)
    spec = DatasetSpec()
    out = build_dataset_for_symbol(bars, "NSE_EQ|TEST", spec=spec)
    X, y = x_y_arrays(out, spec=spec, label="regression")
    assert X.shape[0] == len(out)
    assert X.shape[1] == len(spec.feature_columns)
    assert y.shape == (len(out),)
    assert X.dtype == np.float32 and y.dtype == np.float32


def test_x_y_arrays_rejects_unknown_label():
    bars = pd.concat([_make_session(20000 + i, 100) for i in range(60)], ignore_index=True)
    out = build_dataset_for_symbol(bars, "X")
    with pytest.raises(ValueError, match="must be"):
        x_y_arrays(out, label="bogus")
