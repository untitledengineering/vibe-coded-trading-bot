"""Tests for the training pipeline on synthetic data. We don't validate
predictive quality (XGBoost on synthetic noise won't generalise) — just that
the plumbing runs end-to-end, save/load works, and metrics are computed."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.technical import FEATURE_COLUMNS, IST_OFFSET_SECONDS
from src.model.dataset import DatasetSpec, build_dataset_for_symbol
from src.model.infer import latest_metrics_summary, load_model, score
from src.model.train import (
    FinalMetrics,
    _hit_rate,
    _make_folds,
    fit_production_model,
    save_model,
    walk_forward,
)


def _make_session(day_index: int, n_bars: int, rng):
    base = 100 + rng.normal(0, 1)
    closes = base + rng.normal(0, 0.5, size=n_bars).cumsum() * 0.01
    day_start_utc = day_index * 86400 - IST_OFFSET_SECONDS
    minute_ts = [day_start_utc + 555 * 60 + 60 * i for i in range(n_bars)]
    return pd.DataFrame(
        {
            "minute_ts": minute_ts,
            "open": closes,
            "high": closes + 0.2,
            "low": closes - 0.2,
            "close": closes,
            "volume": rng.integers(100, 1000, size=n_bars),
        }
    )


def _make_synthetic_dataset(n_days: int = 200, bars_per_day: int = 100, n_symbols: int = 3):
    rng = np.random.default_rng(seed=7)
    frames = []
    for sym_idx in range(n_symbols):
        symbol_key = f"NSE_EQ|TEST{sym_idx:03d}"
        bars = pd.concat(
            [_make_session(20000 + d, bars_per_day, rng) for d in range(n_days)],
            ignore_index=True,
        )
        frames.append(build_dataset_for_symbol(bars, symbol_key))
    return pd.concat(frames, ignore_index=True)


# ----- hit rate -----

def test_hit_rate_perfect_alignment_is_one():
    y_true = np.array([0.01, -0.02, 0.03, -0.01])
    y_pred = np.array([0.5, -0.5, 0.5, -0.5])
    assert _hit_rate(y_true, y_pred) == 1.0


def test_hit_rate_excludes_zero_truth():
    y_true = np.array([0.0, 0.0, 0.01])
    y_pred = np.array([1.0, -1.0, 1.0])
    # The two zero-truth rows are excluded; the one remaining row matches.
    assert _hit_rate(y_true, y_pred) == 1.0


def test_hit_rate_random_alignment_is_near_half():
    rng = np.random.default_rng(seed=0)
    y_true = rng.normal(size=10_000)
    y_pred = rng.normal(size=10_000)
    assert 0.45 < _hit_rate(y_true, y_pred) < 0.55


# ----- fold construction -----

def test_make_folds_emits_correct_count_when_history_long():
    # 9 months of data; 3 30-day test windows should fit comfortably.
    one_day = 86400
    ts = np.arange(0, 9 * 30 * one_day, 60)
    folds = _make_folds(ts, n_folds=3, train_min_days=90)
    assert len(folds) == 3
    # Folds are emitted in chronological order; each test_end is later than the prior.
    for i in range(1, len(folds)):
        assert folds[i][1] > folds[i - 1][1]


def test_make_folds_collapses_when_history_short():
    one_day = 86400
    ts = np.arange(0, 60 * one_day, 60)  # 60 days
    folds = _make_folds(ts, n_folds=3, train_min_days=90)
    # 60 days < 90 day train minimum -> single-fold fallback.
    assert len(folds) == 1


# ----- end-to-end training + persistence -----

def test_walk_forward_runs_and_produces_metrics(tmp_path, mocker):
    df = _make_synthetic_dataset(n_days=200, bars_per_day=80, n_symbols=2)
    spec = DatasetSpec()
    folds = walk_forward(df, spec=spec, n_folds=2, num_rounds=20, early_stopping=5)
    assert len(folds) >= 1
    for m in folds:
        assert m.train_rows > 0
        assert m.test_rows > 0
        assert m.test_rmse >= 0
        # corr can be any sign on synthetic noise.
        assert isinstance(m.test_corr, float)


def test_save_and_load_model_round_trip(tmp_path):
    df = _make_synthetic_dataset(n_days=150, bars_per_day=80, n_symbols=2)
    spec = DatasetSpec()
    booster = fit_production_model(df, spec=spec, num_rounds=20)
    model_path = tmp_path / "m.json"
    metrics_path = tmp_path / "metrics.json"
    save_model(
        booster,
        fold_metrics=[],
        final_metrics=FinalMetrics(
            rows=len(df),
            rounds=20,
            features=list(spec.feature_columns),
            label_column=spec.label_regression_column,
        ),
        spec=spec,
        model_path=model_path,
        metrics_path=metrics_path,
    )
    assert model_path.exists()
    assert metrics_path.exists()
    raw = json.loads(metrics_path.read_text())
    assert raw["spec"]["feature_columns"] == list(spec.feature_columns)

    artifact = load_model(model_path=model_path, metrics_path=metrics_path)
    assert artifact.feature_columns == list(spec.feature_columns)
    assert artifact.label_horizon_minutes == spec.label_horizon_minutes

    # Score returns a 1-D array of length n_rows.
    preds = score(artifact, df.head(50))
    assert preds.shape == (50,)


def test_score_rejects_missing_columns(tmp_path):
    df = _make_synthetic_dataset(n_days=120, bars_per_day=60, n_symbols=1)
    spec = DatasetSpec()
    booster = fit_production_model(df, spec=spec, num_rounds=10)
    model_path = tmp_path / "m.json"
    metrics_path = tmp_path / "metrics.json"
    save_model(
        booster,
        [],
        FinalMetrics(
            rows=len(df), rounds=10, features=list(spec.feature_columns),
            label_column=spec.label_regression_column,
        ),
        spec, model_path=model_path, metrics_path=metrics_path,
    )
    artifact = load_model(model_path=model_path, metrics_path=metrics_path)
    bad = df.drop(columns=["rsi_14"]).head(10)
    with pytest.raises(ValueError, match="missing columns"):
        score(artifact, bad)


def test_latest_metrics_summary_aggregates_folds(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({
        "spec": {"feature_columns": list(FEATURE_COLUMNS), "label_horizon_minutes": 15},
        "final": {"rows": 1000, "rounds": 50, "features": list(FEATURE_COLUMNS),
                  "label_column": "fwd_ret_15m"},
        "folds": [
            {"fold": 0, "test_corr": 0.05, "test_hit_rate": 0.52,
             "test_mean_ret_top_decile": 0.001, "test_mean_ret_bottom_decile": -0.001,
             "train_rows": 100, "test_rows": 10, "train_end_ts": 0, "test_end_ts": 0,
             "test_rmse": 0.01, "boost_rounds_used": 50},
            {"fold": 1, "test_corr": 0.07, "test_hit_rate": 0.54,
             "test_mean_ret_top_decile": 0.002, "test_mean_ret_bottom_decile": -0.0015,
             "train_rows": 100, "test_rows": 10, "train_end_ts": 0, "test_end_ts": 0,
             "test_rmse": 0.01, "boost_rounds_used": 50},
        ],
    }))
    # Need a paired model file so load_model() doesn't choke; smallest valid is empty JSON.
    model_path = tmp_path / "m.json"
    # Use the real save path via a quick model fit.
    df = _make_synthetic_dataset(n_days=60, bars_per_day=40, n_symbols=1)
    spec = DatasetSpec()
    booster = fit_production_model(df, spec=spec, num_rounds=5)
    booster.save_model(str(model_path))

    summary = latest_metrics_summary(load_model(model_path=model_path, metrics_path=metrics_path))
    assert summary["folds"] == 2
    assert 0.05 < summary["mean_test_corr"] < 0.08
    assert summary["mean_hit_rate"] == pytest.approx(0.53, abs=1e-9)
