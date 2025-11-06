"""Model training utilities.

This module exposes a minimal, reusable function to train a LightGBM model
using previously discovered best hyperparameters.

WHY: Keep training small and focused (KISS). We fit on the training split only
and rely on cross-validation during hyperparameter search; no dedicated
validation split is used here.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMRegressor

from src.constants import DEFAULT_RANDOM_STATE, TARGET_COLUMN
from src.models.models import create_lightgbm_regressor
from src.utils import (
    ensure_numeric_columns,
    get_logger,
    load_splits_from_parquet,
    read_best_params,
    split_features_target,
    to_project_relative_path,
)

LOGGER = get_logger(__name__)


def _load_numeric_train(
    parquet_path: Path,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric train split ready for modeling."""

    parquet_path = Path(parquet_path)
    LOGGER.info("Loading dataset splits from %s", parquet_path)
    splits = load_splits_from_parquet(parquet_path)
    train_df = splits[0]
    LOGGER.info("Train split: %d rows", len(train_df))

    X_train_raw, y_train = split_features_target(train_df, target_column)
    # Surface a helpful message when non-numeric columns slip through (e.g., when
    # training on an unexpected pre-encoded artifact). The downstream converter is
    # leak-safe and deterministic.
    non_numeric_cols = [
        c for c in X_train_raw.columns if not pd.api.types.is_numeric_dtype(X_train_raw[c])
    ]
    if non_numeric_cols:
        LOGGER.info(
            "Converting %d non-numeric column(s) to numeric: %s",
            len(non_numeric_cols),
            ", ".join(non_numeric_cols[:10]) + ("..." if len(non_numeric_cols) > 10 else ""),
        )
    X_train = ensure_numeric_columns(X_train_raw)
    LOGGER.info("Train shape: %s", X_train.shape)
    return X_train, y_train


def _configure_lightgbm_model(
    params_json_path: Path,
    random_state: int,
) -> LGBMRegressor:
    """Instantiate a LightGBM regressor with safe hyperparameter overrides."""

    params_json_path = Path(params_json_path)
    LOGGER.info("Loading best hyperparameters from %s", params_json_path)
    best_params = read_best_params(params_json_path)
    LOGGER.info("Best hyperparameters: %s", best_params)

    model = create_lightgbm_regressor(random_state=random_state)
    allowed_keys = set(model.get_params().keys())
    safe_params = {k: v for k, v in best_params.items() if k in allowed_keys}
    safe_params.setdefault("n_jobs", -1)  # WHY: ensure concurrency even if tuning omitted it
    safe_params.setdefault("n_estimators", 1000)
    safe_params.setdefault("verbosity", -1)
    model.set_params(**safe_params)
    return model


def _fit_lightgbm_model(
    model: LGBMRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    """Train LightGBM on the full training split without a separate validation set."""

    LOGGER.info("Fitting LightGBM on train split (no dedicated validation)")
    model.fit(X_train, y_train)


def train_lightgbm_with_best_params(
    parquet_path: Path,
    params_json_path: Path,
    *,
    target_column: str = TARGET_COLUMN,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> LGBMRegressor:
    """Train a LightGBM regressor using best hyperparameters.

    WHY: We train on the training split only; model selection and early stopping
    are handled during cross-validation in hyperparameter optimization.

    Args:
        parquet_path: Path to the concatenated Parquet file that contains the
            pre-split encoded dataset with a 'split' column.
        params_json_path: Path to the JSON file containing a top-level key
            "best_params" with hyperparameters to apply.
        target_column: Name of the target column to predict. Defaults to
            "weight-(kg)" as specified in the project.
        random_state: Seed to ensure deterministic behavior across runs.

    Returns:
        A fitted LGBMRegressor instance.

    Raises:
        FileNotFoundError: If the JSON params file does not exist.
        KeyError: If the JSON does not contain the expected keys.
    """
    parquet_path = Path(parquet_path)
    params_json_path = Path(params_json_path)
    X_train, y_train = _load_numeric_train(parquet_path, target_column)
    model = _configure_lightgbm_model(params_json_path, random_state)
    _fit_lightgbm_model(model, X_train, y_train)
    return model


def save_lightgbm_model(
    model: LGBMRegressor,
    model_path: str,
) -> str:
    """Save a trained LightGBM model to disk in joblib format.

    WHY: We standardized the project on LightGBM; keeping the contract minimal
    and explicit avoids ambiguity and simplifies maintenance.

    Args:
        model: A fitted LGBMRegressor instance.
        model_path: Destination file path. If no extension is provided, a
            ".joblib" extension is appended.

    Returns:
        Path to the saved model file relative to the project root when it lives
        inside the repository, otherwise the absolute path.
    """
    if not isinstance(model, LGBMRegressor):
        raise TypeError("model must be LGBMRegressor")

    path_obj = Path(model_path).expanduser()
    if path_obj.suffix.lower() not in {".joblib", ".pkl"}:
        path_obj = path_obj.with_suffix(".joblib")

    if path_obj.parent:
        path_obj.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path_obj)
    return str(to_project_relative_path(path_obj))
