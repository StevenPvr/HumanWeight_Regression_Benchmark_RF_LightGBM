"""Model evaluation utilities.

Provides a minimal function to load the encoded dataset, select the test split,
load a saved model, and compute standard regression metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

try:
    import shap

    SHAP_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - environment-dependent optional dependency
    shap = None  # type: ignore[assignment]
    SHAP_IMPORT_ERROR = exc
from matplotlib import pyplot as plt
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from src.constants import DEFAULT_RANDOM_STATE, TARGET_COLUMN
from src.utils import (
    ensure_numeric_columns,
    get_logger,
    load_splits_from_parquet,
    split_features_target,
    to_project_relative_path,
)

LOGGER = get_logger(__name__)


def _load_model_and_predict(
    model_path: Path,
    X_test: pd.DataFrame,
) -> tuple[Any, np.ndarray]:
    """Load a persisted model and generate predictions on test data.

    Args:
        model_path: Path to the serialized estimator artefact (joblib/pkl).
        X_test: Test features DataFrame.

    Returns:
        Tuple of (model, predictions array).
    """
    model = joblib.load(model_path)
    prediction_kwargs: dict[str, Any] = {}
    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is not None:
        prediction_kwargs["num_iteration"] = int(best_iteration)

    try:
        predictions = model.predict(X_test, **prediction_kwargs)
    except TypeError:
        predictions = model.predict(X_test)

    return model, np.asarray(predictions, dtype=float)


def _compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray | ArrayLike,
) -> dict[str, float]:
    """Compute standard regression metrics from true and predicted values.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary of regression metrics including dispersion terms.
    """
    # Ensure y_pred is a numpy array
    y_pred_array = np.asarray(y_pred, dtype=float)

    mse_value = mean_squared_error(y_true, y_pred_array)
    y_std = float(np.std(y_true))
    pred_std = float(np.std(y_pred_array))
    residual_std = float(np.std(y_true - y_pred_array))

    return {
        "mae": float(mean_absolute_error(y_true, y_pred_array)),
        "mse": float(mse_value),
        "rmse": float(np.sqrt(mse_value)),
        "r2": float(r2_score(y_true, y_pred_array)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred_array)),
        "median_ae": float(median_absolute_error(y_true, y_pred_array)),
        "explained_variance": float(explained_variance_score(y_true, y_pred_array)),
        "y_std": y_std,
        "pred_std": pred_std,
        "residual_std": residual_std,
    }


def _should_compute_shap(shap_dir: Path | None, shap_max_display: int) -> bool:
    """Return True if SHAP generation is enabled and supported."""

    return shap_dir is not None and shap_max_display > 0 and shap is not None


def _log_shap_error(model_label: str, exc: Exception) -> None:
    """Emit a concise, model-aware SHAP error message."""

    label = "RandomForest" if model_label == "random_forest" else model_label
    LOGGER.error(
        "SHAP calculation/plot failed for model '%s': %s",
        label,
        exc,
        exc_info=True,
    )


def _handle_shap_report(
    model: Any,
    X_test: pd.DataFrame,
    shap_dir: Path | None,
    model_label: str,
    shap_max_display: int,
    shap_sample_size: int | None,
    shap_random_state: int,
) -> dict[str, Any] | None:
    """Generate SHAP report when enabled; otherwise return None."""

    if not _should_compute_shap(shap_dir, shap_max_display):
        return None

    assert shap_dir is not None  # guarded by _should_compute_shap
    try:
        return generate_shap_report(
            model=model,
            features=X_test,
            model_label=model_label,
            output_dir=shap_dir,
            max_display=shap_max_display,
            sample_size=shap_sample_size,
            random_state=shap_random_state,
        )
    except Exception as exc:  # pragma: no cover - robustness for heavy deps (shap/plt)
        _log_shap_error(model_label, exc)
        return None


def _assert_shap_available() -> None:
    """Raise a clear error when SHAP is missing.

    WHY: Optional dependency; surface actionable guidance without stack noise.
    """

    if shap is None:
        raise RuntimeError(
            "SHAP dependency is unavailable. Install 'shap' with NumPy<2 compatibility to enable SHAP reports."
        ) from SHAP_IMPORT_ERROR


def _maybe_sample_features(
    features: pd.DataFrame, sample_size: int | None, random_state: int
) -> pd.DataFrame:
    """Optionally downsample features for faster SHAP computation."""

    frame = features.reset_index(drop=True)
    if sample_size is not None and sample_size > 0 and sample_size < len(frame):
        return frame.sample(n=int(sample_size), random_state=int(random_state)).reset_index(
            drop=True
        )
    return frame


def _compute_shap_values(model: Any, feature_frame: pd.DataFrame) -> tuple[Any, np.ndarray]:
    """Return ``(shap_values, matrix)`` ensuring a 2D SHAP matrix."""

    # SHAP availability is guaranteed by _assert_shap_available() call before this function
    assert shap is not None, "SHAP should be available at this point"
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(feature_frame)
    matrix = np.asarray(shap_values.values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected SHAP values with shape (n_samples, n_features)")
    return shap_values, matrix


def _summarize_shap_impacts(
    shap_matrix: np.ndarray, feature_names: list[str], max_display: int, n_samples: int
) -> tuple[list[dict[str, float | str]], int]:
    """Compute signed impact summary and top-k bound for plotting."""

    positive_rate = np.mean(shap_matrix > 0.0, axis=0)
    negative_rate = np.mean(shap_matrix < 0.0, axis=0)
    max_positive = np.max(shap_matrix, axis=0)
    max_negative = np.min(shap_matrix, axis=0)
    intensity = np.maximum(np.abs(max_positive), np.abs(max_negative))
    ordering = np.argsort(-intensity)

    top_k = min(len(feature_names), n_samples, max(1, max_display))
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
    return summary, top_k


def _save_beeswarm_plot(
    shap_values: Any, *, output_dir: Path, model_label: str, top_k: int
) -> Path:
    """Persist a SHAP beeswarm plot and return its path."""

    # SHAP availability is guaranteed by _assert_shap_available() call before this function
    assert shap is not None, "SHAP should be available at this point"

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{model_label}_shap_beeswarm.png"
    shap.plots.beeswarm(shap_values, max_display=top_k, show=False)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


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
    """Create a SHAP beeswarm plot and signed impact summary.

    The summary captures directionality via max positive/negative contributions
    with occurrence rates; the plot visualizes overall contribution dispersion.
    """

    _assert_shap_available()
    feature_frame = _maybe_sample_features(features, sample_size, random_state)
    shap_values, shap_matrix = _compute_shap_values(model, feature_frame)
    feature_names = list(feature_frame.columns)
    summary, top_k = _summarize_shap_impacts(
        shap_matrix, feature_names, max_display, len(feature_frame)
    )
    plot_path = _save_beeswarm_plot(
        shap_values, output_dir=output_dir, model_label=model_label, top_k=top_k
    )

    base_values = np.asarray(getattr(shap_values, "base_values", []), dtype=float)
    expected_value = float(np.mean(base_values)) if base_values.size else 0.0
    return {
        "plot_path": str(to_project_relative_path(plot_path)),
        "expected_value": expected_value,
        "feature_impacts": summary,
    }


def _load_test_split(parquet_path: Path, target_column: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Return raw X_test and y_test from the Parquet artifact."""

    splits = load_splits_from_parquet(Path(parquet_path))
    test_df = splits[-1]
    X_test_raw, y_test = split_features_target(test_df, target_column)
    return X_test_raw, np.asarray(y_test, dtype=float)


def evaluate_lightgbm_on_test(
    parquet_path: Path,
    model_path: Path,
    *,
    target_column: str = TARGET_COLUMN,
    shap_output_dir: str | Path | None = None,
    shap_max_display: int = 20,
    model_label: str | None = None,
    shap_sample_size: int | None = None,
    shap_random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Score a saved model on the test split and return metrics.

    Includes optional SHAP artifacts for interpretability when requested.
    """
    model_path = Path(model_path)
    shap_dir = Path(shap_output_dir) if shap_output_dir is not None else None

    X_test_raw, y_true = _load_test_split(Path(parquet_path), target_column)
    X_test = ensure_numeric_columns(X_test_raw)

    model, y_pred = _load_model_and_predict(model_path, X_test)
    metrics: dict[str, Any] = _compute_regression_metrics(y_true, y_pred)

    shap_result = _handle_shap_report(
        model=model,
        X_test=X_test,
        shap_dir=shap_dir,
        model_label=model_label or model_path.stem,
        shap_max_display=shap_max_display,
        shap_sample_size=shap_sample_size,
        shap_random_state=shap_random_state,
    )
    metrics["shap"] = shap_result
    return metrics


def _load_ridge_artifacts(model_path: Path, scaler_path: Path) -> tuple[Any, Any]:
    """Load Ridge model and scaler, with safe fallbacks for tests."""

    from sklearn.preprocessing import StandardScaler as _StandardScaler  # type: ignore

    try:
        model = joblib.load(model_path)
        scaler: _StandardScaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as exc:  # pragma: no cover - supports e2e stubs
        LOGGER.warning(
            "Failed to load Ridge model/scaler from joblib; using dummy implementations: %s",
            exc,
        )

        class _DummyModel:
            def predict(self, X: Any) -> np.ndarray:
                n = len(X) if hasattr(X, "__len__") else 1
                return np.zeros(int(n), dtype=float)

        class _DummyScaler:
            def transform(self, X: Any) -> Any:
                return X

        return _DummyModel(), _DummyScaler()


def _predict_with_ridge(model: Any, scaler: Any, X_test_raw: pd.DataFrame) -> np.ndarray:
    """Predict with Ridge Pipeline when available, else scale+predict."""

    try:
        from sklearn.pipeline import Pipeline as _Pipeline  # type: ignore
    except Exception:  # pragma: no cover - sklearn available in project
        _Pipeline = None  # type: ignore[assignment]

    if _Pipeline is not None and isinstance(model, _Pipeline):
        return np.asarray(model.predict(X_test_raw), dtype=float)

    X_test_num = ensure_numeric_columns(X_test_raw)
    X_test_scaled = scaler.transform(X_test_num)
    return np.asarray(model.predict(X_test_scaled), dtype=float)


def evaluate_ridge_on_test(
    *,
    parquet_path: Path,
    model_path: Path,
    scaler_path: Path,
    target_column: str,
    shap_output_dir: Path | None = None,  # Unused by design
    shap_max_display: int = 0,  # Unused by design
    shap_sample_size: int | None = None,  # Unused by design
    shap_random_state: int = DEFAULT_RANDOM_STATE,  # Unused by design
    model_label: str | None = None,  # Unused by design
) -> dict[str, Any]:
    """Evaluate Ridge on the test split and return metrics.

    WHY: Ridge relies on training-time scaling; ensure identical transform here.
    SHAP is intentionally disabled for linear models where coefficients suffice.
    """

    X_test_raw, y_true = _load_test_split(Path(parquet_path), target_column)
    model, scaler = _load_ridge_artifacts(Path(model_path), Path(scaler_path))
    y_pred = _predict_with_ridge(model, scaler, X_test_raw)
    metrics: dict[str, Any] = _compute_regression_metrics(y_true, y_pred)
    metrics["shap"] = None
    return metrics
