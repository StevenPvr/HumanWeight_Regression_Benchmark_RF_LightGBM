"""Model evaluation utilities.

Provides a minimal function to load the encoded dataset, select the test split,
load a saved model, and compute standard regression metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    explained_variance_score,
)

from src.constants import TARGET_COLUMN, DEFAULT_RANDOM_STATE
from src.utils import get_logger, load_splits_from_parquet, split_features_target, ensure_numeric_columns

LOGGER = get_logger(__name__)


def generate_shap_report(
    *,
    model: Any,
    features: pd.DataFrame,
    model_label: str,
    output_dir: Path,
    max_display: int,
    sample_size: int | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Create a SHAP beeswarm plot plus directional impact summary for a model.

    WHY: Tree models can hide feature directionality; storing max positive/negative
    contributions highlights how each feature pushes predictions without relying on
    mean absolute values, matching stakeholders' need for signed impact intuition.
    """

    feature_frame = features.reset_index(drop=True)
    if sample_size is not None and sample_size > 0 and sample_size < len(feature_frame):
        feature_frame = feature_frame.sample(
            n=int(sample_size),
            random_state=int(random_state),
        ).reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(feature_frame)
    shap_matrix = np.asarray(shap_values.values, dtype=float)
    if shap_matrix.ndim != 2:
        raise ValueError("Expected SHAP values with shape (n_samples, n_features)")

    positive_rate = np.mean(shap_matrix > 0.0, axis=0)
    negative_rate = np.mean(shap_matrix < 0.0, axis=0)
    max_positive = np.max(shap_matrix, axis=0)
    max_negative = np.min(shap_matrix, axis=0)
    intensity = np.maximum(np.abs(max_positive), np.abs(max_negative))
    ordering = np.argsort(-intensity)
    feature_names = list(feature_frame.columns)

    top_k = min(len(feature_names), len(feature_frame), max(1, max_display))
    summary = [
        {
            "feature": feature_names[idx],
            "max_positive": float(max_positive[idx]),
            "max_negative": float(max_negative[idx]),
            "positive_rate": float(positive_rate[idx]),
            "negative_rate": float(negative_rate[idx]),
            "max_intensity": float(intensity[idx]),
        }
        for idx in ordering[:top_k]
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{model_label}_shap_beeswarm.png"
    shap.plots.beeswarm(shap_values, max_display=top_k, show=False)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    base_values = np.asarray(shap_values.base_values, dtype=float)
    expected_value = float(np.mean(base_values)) if base_values.size else 0.0
    return {
        "plot_path": str(plot_path),
        "expected_value": expected_value,
        "feature_impacts": summary,
    }


def evaluate_lightgbm_on_test(
    parquet_path: str,
    model_path: str,
    *,
    target_column: str = TARGET_COLUMN,
    shap_output_dir: str | Path | None = None,
    shap_max_display: int = 20,
    model_label: str | None = None,
    shap_sample_size: int | None = None,
    shap_random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Evaluate a persisted model on the held-out test split.

    WHY: Keeps evaluation isolated from training artefacts so the test data
    remains untouched until final scoring, ensuring comparable metrics. We also
    report dispersion terms that disambiguate sources of variance:
      - y_std: spread of ground-truth labels (dataset property)
      - pred_std: spread of predictions (model behaviour)
      - residual_std: spread of errors (useful to compare models)

    Args:
        parquet_path: Path to the concatenated Parquet file containing the splits.
        model_path: Path to the serialized estimator artefact (joblib/pkl).
        target_column: Name of the regression target column.

    Returns:
        Mapping of standard regression metrics summarising the test performance.
        Includes a ``shap`` key when SHAP artefacts were generated.
    """
    _, _, test_df = load_splits_from_parquet(parquet_path)
    X_test_raw, y_test = split_features_target(test_df, target_column)
    X_test = ensure_numeric_columns(X_test_raw)

    model = joblib.load(model_path)
    prediction_kwargs: dict[str, Any] = {}
    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is not None:
        prediction_kwargs["num_iteration"] = int(best_iteration)

    try:
        predictions = model.predict(X_test, **prediction_kwargs)
    except TypeError:
        predictions = model.predict(X_test)

    y_true = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(predictions, dtype=float)

    mse_value = mean_squared_error(y_true, y_pred)
    # WHY: Expose dataset variance and model-induced dispersions separately
    y_std = float(np.std(y_true))
    pred_std = float(np.std(y_pred))
    residual_std = float(np.std(y_true - y_pred))
    metrics: dict[str, Any] = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse_value),
        "rmse": float(np.sqrt(mse_value)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
        "y_std": y_std,
        "pred_std": pred_std,
        "residual_std": residual_std,
    }
    shap_enabled = shap_output_dir is not None and shap_max_display > 0
    if shap_enabled:
        shap_dir_input = cast(str | Path, shap_output_dir)
        try:
            metrics["shap"] = generate_shap_report(
                model=model,
                features=X_test,
                model_label=model_label or Path(model_path).stem,
                output_dir=Path(shap_dir_input),
                max_display=shap_max_display,
                sample_size=shap_sample_size,
                random_state=shap_random_state,
            )
        except Exception as exc:  # pragma: no cover - robustness for heavy deps (shap/plt)
            # WHY: SHAP plotting can fail for some model types or environments
            # (missing display, unexpected explainer behaviour). For RandomForest
            # we want to surface an explicit error and continue without failing
            # the whole evaluation workflow.
            if model_label == "random_forest":
                LOGGER.error(
                    "SHAP calculation/plot failed for RandomForest model '%s': %s",
                    model_label,
                    exc,
                    exc_info=True,
                )
            else:
                LOGGER.error(
                    "SHAP calculation/plot failed for model '%s': %s",
                    model_label or Path(model_path).stem,
                    exc,
                    exc_info=True,
                )
            metrics["shap"] = None
    else:
        metrics["shap"] = None
    return metrics
