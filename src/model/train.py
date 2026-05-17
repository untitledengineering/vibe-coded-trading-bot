"""Walk-forward training of the intraday return regressor (XGBoost).

Two outputs:
    1. Out-of-sample evaluation metrics across multiple time-respecting folds,
       to tell us whether the signal generalises.
    2. A "production" model fit on the entire backfilled history, written to
       data/model_v1.json. The decision engine in Sprint 3 loads this artifact.

CLI:
    python -m src.model.train                     # train on full universe, default folds
    python -m src.model.train --symbols RELIANCE  # restrict for smoke testing
    python -m src.model.train --quick             # 1 fold, fewer rounds (CI / sanity)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from src.model.dataset import (
    DatasetSpec,
    assemble_full_dataset,
    time_split,
    x_y_arrays,
)
from src.model.labels import DEFAULT_HORIZON_MINUTES
from src.utils.logger import logger

MODEL_PATH = Path("data/model_v1.json")
METRICS_PATH = Path("data/model_v1_metrics.json")


def paths_for_name(name: str) -> tuple[Path, Path]:
    """Conventional artifact paths for a model name (e.g. 'v1', 'v2_h30')."""
    return Path(f"data/model_{name}.json"), Path(f"data/model_{name}_metrics.json")

DEFAULT_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 50,
    "verbosity": 0,
    "nthread": 0,  # 0 = use all cores
}
DEFAULT_NUM_BOOST_ROUNDS = 400
DEFAULT_EARLY_STOPPING = 25
QUICK_NUM_BOOST_ROUNDS = 60


@dataclass
class FoldMetrics:
    fold: int
    train_rows: int
    test_rows: int
    train_end_ts: int
    test_end_ts: int
    test_rmse: float
    test_corr: float
    test_hit_rate: float
    test_mean_ret_top_decile: float
    test_mean_ret_bottom_decile: float
    boost_rounds_used: int


@dataclass
class FinalMetrics:
    rows: int
    rounds: int
    features: List[str]
    label_column: str


def _hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of rows where sign(y_pred) == sign(y_true). Excludes y_true == 0 ties."""
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(y_pred[mask]) == np.sign(y_true[mask])))


def _decile_returns(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Mean realised return in the top and bottom prediction deciles.
    Top decile should outperform; bottom should underperform. The spread is the
    'edge' the model would produce if we naively traded the deciles."""
    n = len(y_pred)
    if n < 10:
        return float("nan"), float("nan")
    order = np.argsort(y_pred)
    cut = max(1, n // 10)
    bot = order[:cut]
    top = order[-cut:]
    return float(np.mean(y_true[top])), float(np.mean(y_true[bot]))


def _make_folds(
    sorted_ts: np.ndarray,
    n_folds: int,
    train_min_days: int = 90,
) -> List[tuple[int, int]]:
    """Return list of (train_end_ts, test_end_ts). Each fold uses an expanding
    train window with a 30-day test window appended."""
    if len(sorted_ts) == 0:
        return []
    earliest = int(sorted_ts[0])
    latest = int(sorted_ts[-1])
    one_day = 86400
    test_window = 30 * one_day
    train_min = train_min_days * one_day

    available_days = (latest - earliest) // one_day
    if available_days < train_min_days + 30:
        # Not enough history for the requested fold count; emit a single fold.
        return [(earliest + train_min, latest)]

    folds: List[tuple[int, int]] = []
    # Test windows step back from the latest date so the most recent OOS is captured.
    test_end = latest
    while len(folds) < n_folds and test_end > earliest + train_min + test_window:
        train_end = test_end - test_window
        folds.append((train_end, test_end))
        test_end = train_end
    folds.reverse()
    return folds


def _train_one_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict,
    num_rounds: int,
    early_stopping: int,
    spec: DatasetSpec,
) -> tuple[xgb.Booster, dict]:
    X_train, y_train = x_y_arrays(train_df, spec=spec, label="regression")
    X_test, y_test = x_y_arrays(test_df, spec=spec, label="regression")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(spec.feature_columns))
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(spec.feature_columns))

    evals_result: dict = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=early_stopping,
        evals_result=evals_result,
        verbose_eval=False,
    )
    return booster, evals_result


def walk_forward(
    df: pd.DataFrame,
    spec: DatasetSpec,
    n_folds: int = 3,
    num_rounds: int = DEFAULT_NUM_BOOST_ROUNDS,
    early_stopping: int = DEFAULT_EARLY_STOPPING,
    params: Optional[dict] = None,
) -> List[FoldMetrics]:
    """Run walk-forward training and return per-fold OOS metrics. The fold
    boosters are discarded — the production model is fit separately on all data."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    sorted_ts = df["minute_ts"].sort_values().to_numpy()
    folds = _make_folds(sorted_ts, n_folds=n_folds)
    if not folds:
        raise RuntimeError("Not enough data to construct a single fold.")

    out: List[FoldMetrics] = []
    for i, (train_end, test_end) in enumerate(folds):
        train_df, test_df = time_split(df, train_end_ts=train_end, test_end_ts=test_end)
        if len(train_df) < 10_000 or len(test_df) < 1_000:
            logger.warning(
                f"Fold {i}: skipping (train={len(train_df)}, test={len(test_df)} too small)"
            )
            continue
        logger.info(
            f"Fold {i}: train_rows={len(train_df):,} test_rows={len(test_df):,} "
            f"train_end={pd.to_datetime(train_end, unit='s')} "
            f"test_end={pd.to_datetime(test_end, unit='s')}"
        )
        booster, _ = _train_one_fold(
            train_df, test_df, params, num_rounds, early_stopping, spec
        )
        X_test, y_test = x_y_arrays(test_df, spec=spec, label="regression")
        dtest = xgb.DMatrix(X_test, feature_names=list(spec.feature_columns))
        # Use best_iteration explicitly so we don't include rounds past early-stop.
        best_iter = booster.best_iteration if booster.best_iteration is not None else num_rounds - 1
        y_pred = booster.predict(dtest, iteration_range=(0, best_iter + 1))

        rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
        # Pearson correlation; guard against zero-variance prediction.
        if np.std(y_pred) == 0 or np.std(y_test) == 0:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(y_pred, y_test)[0, 1])
        hit = _hit_rate(y_test, y_pred)
        top_dec, bot_dec = _decile_returns(y_test, y_pred)
        m = FoldMetrics(
            fold=i,
            train_rows=len(train_df),
            test_rows=len(test_df),
            train_end_ts=train_end,
            test_end_ts=test_end,
            test_rmse=rmse,
            test_corr=corr if not math.isnan(corr) else 0.0,
            test_hit_rate=hit if not math.isnan(hit) else 0.0,
            test_mean_ret_top_decile=top_dec if not math.isnan(top_dec) else 0.0,
            test_mean_ret_bottom_decile=bot_dec if not math.isnan(bot_dec) else 0.0,
            boost_rounds_used=best_iter + 1,
        )
        logger.info(
            f"  -> rmse={m.test_rmse:.6f} corr={m.test_corr:.4f} hit={m.test_hit_rate:.3f} "
            f"top={m.test_mean_ret_top_decile:+.5f} bot={m.test_mean_ret_bottom_decile:+.5f}"
        )
        out.append(m)
    return out


def fit_production_model(
    df: pd.DataFrame,
    spec: DatasetSpec,
    num_rounds: int,
    params: Optional[dict] = None,
) -> xgb.Booster:
    """Fit one final model on ALL data. No held-out test — the walk-forward
    metrics are how we judge fit quality."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    X, y = x_y_arrays(df, spec=spec, label="regression")
    dtrain = xgb.DMatrix(X, label=y, feature_names=list(spec.feature_columns))
    booster = xgb.train(params, dtrain, num_boost_round=num_rounds, verbose_eval=False)
    return booster


def save_model(
    booster: xgb.Booster,
    fold_metrics: List[FoldMetrics],
    final_metrics: FinalMetrics,
    spec: DatasetSpec,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_path))
    metrics = {
        "spec": {
            "feature_columns": list(spec.feature_columns),
            "label_horizon_minutes": spec.label_horizon_minutes,
        },
        "final": asdict(final_metrics),
        "folds": [asdict(m) for m in fold_metrics],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Saved model to {model_path} and metrics to {metrics_path}")


def _print_summary(fold_metrics: List[FoldMetrics]) -> None:
    if not fold_metrics:
        print("No folds completed.")
        return
    print()
    print("Walk-forward results:")
    print(f"  {'fold':>4}  {'train':>9}  {'test':>9}  {'rmse':>8}  {'corr':>6}  "
          f"{'hit%':>6}  {'top dec':>10}  {'bot dec':>10}")
    for m in fold_metrics:
        print(
            f"  {m.fold:>4}  {m.train_rows:>9,}  {m.test_rows:>9,}  "
            f"{m.test_rmse:>8.5f}  {m.test_corr:>+6.3f}  "
            f"{m.test_hit_rate*100:>5.2f}%  "
            f"{m.test_mean_ret_top_decile:>+10.5f}  {m.test_mean_ret_bottom_decile:>+10.5f}"
        )
    spread = np.mean(
        [m.test_mean_ret_top_decile - m.test_mean_ret_bottom_decile for m in fold_metrics]
    )
    mean_corr = np.mean([m.test_corr for m in fold_metrics])
    print()
    print(f"Mean test corr:        {mean_corr:+.4f}")
    print(f"Mean top-bot spread:   {spread:+.5f}  ({spread*100:+.3f} pp per trade)")


def main():
    parser = argparse.ArgumentParser(description="Train the intraday return regressor.")
    parser.add_argument("--symbols", nargs="*", help="Restrict universe (smoke testing)")
    parser.add_argument("--folds", type=int, default=3, help="Number of walk-forward folds")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fewer boosting rounds. Useful for sanity-checking the pipeline.",
    )
    parser.add_argument(
        "--name",
        default="v1",
        help="Artifact name. Outputs land at data/model_<name>.json + _metrics.json.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON_MINUTES,
        help="Forward-return label horizon in minutes (default 15).",
    )
    args = parser.parse_args()

    spec = DatasetSpec(label_horizon_minutes=args.horizon)
    model_path, metrics_path = paths_for_name(args.name)
    logger.info(
        f"Training model name='{args.name}' horizon={args.horizon}m -> {model_path}"
    )
    started = time.monotonic()
    logger.info("Assembling dataset...")
    df = assemble_full_dataset(spec=spec, symbols=args.symbols)
    if df.empty:
        raise SystemExit("Empty dataset. Run the historical backfill first.")
    logger.info(f"Dataset built in {time.monotonic() - started:.1f}s. Rows: {len(df):,}")

    rounds = QUICK_NUM_BOOST_ROUNDS if args.quick else DEFAULT_NUM_BOOST_ROUNDS
    fold_metrics = walk_forward(df, spec=spec, n_folds=args.folds, num_rounds=rounds)
    _print_summary(fold_metrics)

    # Use a sensible round count for the production fit. If walk-forward picked a
    # smaller best_iteration on average, use that; otherwise stay at the cap.
    if fold_metrics:
        avg_best = int(np.mean([m.boost_rounds_used for m in fold_metrics]))
        production_rounds = max(50, min(rounds, avg_best))
    else:
        production_rounds = rounds

    logger.info(f"Fitting production model on full dataset ({production_rounds} rounds)...")
    booster = fit_production_model(df, spec=spec, num_rounds=production_rounds)
    final_metrics = FinalMetrics(
        rows=len(df),
        rounds=production_rounds,
        features=list(spec.feature_columns),
        label_column=spec.label_regression_column,
    )
    save_model(booster, fold_metrics, final_metrics, spec,
               model_path=model_path, metrics_path=metrics_path)
    logger.info(f"Done. Total time: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
