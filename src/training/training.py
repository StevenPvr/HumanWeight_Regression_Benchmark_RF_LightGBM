"""Model training utilities.

This module exposes a minimal, reusable function to train a LightGBM model
using previously discovered best hyperparameters.

WHY: Keep training small and focused (KISS). We fit on the training split only
and use the validation split with LightGBM early stopping based on MSE.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import lightgbm as lgb  # type: ignore
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

from src.utils import (
    load_splits_from_parquet,
    split_features_target,
    ensure_numeric_columns,
    read_best_params,
    to_project_relative_path,
)
from src.models.models import create_lightgbm_regressor
from src.constants import (
    DEFAULT_RANDOM_STATE,
    HYPEROPT_LGBM_EARLY_STOPPING_ROUNDS,
    TARGET_COLUMN,
)


logger = logging.getLogger(__name__)


def _load_numeric_splits(
    parquet_path: str,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return numeric train/validation splits ready for modeling."""

    logger.info("Loading dataset splits from %s", parquet_path)
    train_df, val_df, _ = load_splits_from_parquet(parquet_path)
    logger.info("Train split: %d rows | Val split: %d rows", len(train_df), len(val_df))

    X_train_raw, y_train = split_features_target(train_df, target_column)
    X_val_raw, y_val = split_features_target(val_df, target_column)
    X_train = ensure_numeric_columns(X_train_raw)
    X_val = ensure_numeric_columns(X_val_raw)
    logger.info("Train shape: %s | Val shape: %s", X_train.shape, X_val.shape)
    return X_train, y_train, X_val, y_val


def _configure_lightgbm_model(
    params_json_path: str,
    random_state: int,
) -> LGBMRegressor:
    """Instantiate a LightGBM regressor with safe hyperparameter overrides."""

    logger.info("Loading best hyperparameters from %s", params_json_path)
    best_params = read_best_params(params_json_path)
    logger.info("Best hyperparameters: %s", best_params)

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
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """Train LightGBM with early stopping and return validation MSE."""

    logger.info("Fitting LightGBM with early stopping")
    callbacks = [
        lgb.early_stopping(stopping_rounds=HYPEROPT_LGBM_EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(period=0),
    ]
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l2",
        callbacks=callbacks,
    )
    best_iter = getattr(model, "best_iteration_", None)
    predictions = (
        model.predict(X_val, num_iteration=best_iter)
        if best_iter is not None
        else model.predict(X_val)
    )
    mse = float(mean_squared_error(np.asarray(y_val), np.asarray(predictions)))
    logger.info("Validation MSE (best iteration): %.6f", mse)
    return mse


def train_lightgbm_with_best_params(
    parquet_path: str,
    params_json_path: str,
    *,
    target_column: str = TARGET_COLUMN,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> LGBMRegressor:
    """Train a LightGBM regressor using best hyperparameters with early stopping.

    WHY: We train on the training split and use the validation split with
    LightGBM early stopping to find the best iteration without overfitting.

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
    X_train, y_train, X_val, y_val = _load_numeric_splits(parquet_path, target_column)
    model = _configure_lightgbm_model(params_json_path, random_state)
    _fit_lightgbm_model(model, X_train, y_train, X_val, y_val)
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
