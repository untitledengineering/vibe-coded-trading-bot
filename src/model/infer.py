"""Inference wrapper for the trained model.

Used by the decision engine in Sprint 3. Loads model_v1.json on construction
and scores a DataFrame of features. The feature column order MUST match what
the model was trained on — we enforce that by reading the spec saved alongside
the model and rejecting frames whose columns don't line up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from src.model.train import METRICS_PATH, MODEL_PATH
from src.utils.logger import logger


@dataclass(frozen=True)
class ModelArtifact:
    booster: xgb.Booster
    feature_columns: List[str]
    label_horizon_minutes: int
    metadata: dict


def load_model(
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> ModelArtifact:
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run `python -m src.model.train` first."
        )
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Model spec missing at {metrics_path}. Re-run training to regenerate."
        )

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    metadata = json.loads(metrics_path.read_text())
    spec = metadata.get("spec", {})
    features = spec.get("feature_columns")
    horizon = spec.get("label_horizon_minutes")
    if not features or not horizon:
        raise ValueError(f"Spec at {metrics_path} is incomplete: {spec}")

    return ModelArtifact(
        booster=booster,
        feature_columns=list(features),
        label_horizon_minutes=int(horizon),
        metadata=metadata,
    )


def score(
    artifact: ModelArtifact,
    feature_frame: pd.DataFrame,
) -> np.ndarray:
    """Predict E[return_horizon] for every row of feature_frame.

    feature_frame must contain (at least) artifact.feature_columns. Any extra
    columns are ignored. The frame must already have NaNs handled by the caller
    — XGBoost tolerates NaN in features, but the decision engine should know
    when it's scoring partial inputs."""
    missing = [c for c in artifact.feature_columns if c not in feature_frame.columns]
    if missing:
        raise ValueError(f"feature_frame missing columns: {missing}")
    X = feature_frame[artifact.feature_columns].to_numpy(dtype=np.float32)
    dmat = xgb.DMatrix(X, feature_names=list(artifact.feature_columns))
    return artifact.booster.predict(dmat)


def latest_metrics_summary(artifact: Optional[ModelArtifact] = None) -> dict:
    """Convenience for the dashboard: pull a small dict of model-health stats
    out of the metrics file. Keeps the dashboard module decoupled from XGBoost."""
    artifact = artifact or load_model()
    folds = artifact.metadata.get("folds", [])
    if not folds:
        return {"trained_on_rows": artifact.metadata.get("final", {}).get("rows"), "folds": 0}
    return {
        "trained_on_rows": artifact.metadata.get("final", {}).get("rows"),
        "folds": len(folds),
        "mean_test_corr": float(np.mean([f["test_corr"] for f in folds])),
        "mean_top_minus_bot_decile_return": float(
            np.mean(
                [
                    f["test_mean_ret_top_decile"] - f["test_mean_ret_bottom_decile"]
                    for f in folds
                ]
            )
        ),
        "mean_hit_rate": float(np.mean([f["test_hit_rate"] for f in folds])),
    }
