from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb  # type: ignore
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import src.utils as _utils
from src.constants import CV_N_SPLITS, DEFAULT_RANDOM_STATE, HYPEROPT_DEFAULT_N_TRIALS
from src.models.models import create_lightgbm_regressor, create_random_forest_regressor


class _SilentLogger:
    """LightGBM logger that discards info and warning messages.

    This object exposes the minimal logging methods used by LightGBM and
    intentionally drops messages. It is used to reduce C++/library log noise
    during automated optimization runs.
    """

    def info(self, msg: str) -> None:  # pragma: no cover - trivial sink
        return None

    def warning(self, msg: str) -> None:  # pragma: no cover - trivial sink
        return None


def _build_cv_dataset(
    parquet_path: Path,
    target_column: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load raw train features/target for cross-validation (no global encoding).

    WHY: Encoders must be fit inside each fold on the training subset only to
    surface unknown categories in the validation subset as all-zero vectors
    (handle_unknown="ignore"). This mirrors production behavior and avoids
    optimistic leakage.
    """
    parquet_path = Path(parquet_path)
    loaded = _utils.load_splits_from_parquet(parquet_path)
    # Support both (train, test) and (train, val, test)
    if isinstance(loaded, tuple) and len(loaded) == 3:  # type: ignore[arg-type]
        train_df = loaded[0]  # type: ignore[index]
    else:
        train_df, _ = loaded  # type: ignore[misc]

    X_train_raw, y_train = _utils.split_features_target(train_df, target_column)
    y_train_arr = np.asarray(y_train)
    return X_train_raw, y_train_arr


def optimize_lightgbm_hyperparameters(
    parquet_path: Path,
    target_column: str,
    *,
    n_trials: int = HYPEROPT_DEFAULT_N_TRIALS,
    random_state: int = DEFAULT_RANDOM_STATE,
    cv_n_splits: int = CV_N_SPLITS,
) -> tuple[dict[str, Any], float, dict[str, float]]:
    """Run Optuna hyperparameter search for a LightGBM regressor.

    Args:
        parquet_path: Path to the Parquet file with pre-split data (train/test).
        target_column: Name of the target column inside the splits.

    Keyword Args:
        n_trials: Number of Optuna trials to execute.
        random_state: RNG seed used for reproducibility (sampler + model factories).
        cv_n_splits: Number of folds for K-fold cross-validation applied to
            the train set.

    Returns:
        A tuple of (best_params, best_value, cv_summary) where `best_params`
        is the dictionary of hyperparameters found by Optuna, `best_value` is
        the cross-validated MSE (float) minimized by the study, and `cv_summary`
        contains a small mapping of cross-validation metrics (e.g. {"cv_mse": ...}).

    Raises:
        RuntimeError: If the Optuna study completes but no best trial value is
            available (should not happen in normal runs).
    """
    X, y = _build_cv_dataset(parquet_path, target_column)

    # Reduce Optuna logger noise.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Silence LightGBM C++ logs.
    lgb.register_logger(_SilentLogger())

    kf = KFold(n_splits=cv_n_splits, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 512),
            "max_depth": trial.suggest_int("max_depth", 3, 18),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 400),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 80.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 40.0),
            "n_jobs": -1,
            "random_state": random_state,
            "verbosity": -1,
        }

        mse_values: list[float] = []

        for train_idx, val_idx in kf.split(X):
            # Fold-wise encoding: fit on train fold only, transform val fold
            X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
            (X_train_enc, X_val_enc), _enc = _utils.fit_label_encoders_on_train(
                X_train_raw,
                [X_val_raw],
            )
            X_train = _utils.ensure_numeric_columns(X_train_enc)
            X_val = _utils.ensure_numeric_columns(X_val_enc)
            y_train, y_val = y[train_idx], y[val_idx]

            model = create_lightgbm_regressor(random_state=random_state)
            allowed = set(model.get_params().keys())
            safe_params = {k: v for k, v in params.items() if k in allowed}
            model.set_params(**safe_params)

            # Fit without an explicit validation set or early stopping; CV provides fold-wise validation.
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            fold_mse = float(mean_squared_error(np.asarray(y_val), np.asarray(y_pred)))
            mse_values.append(fold_mse)

        return float(np.mean(mse_values))

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_value = study.best_trial.value
    if best_value is None:
        raise RuntimeError("Optuna study completed but best_trial.value is None.")

    best_params: dict[str, Any] = study.best_trial.params
    cv_summary = {"cv_mse": float(best_value), "val_mse": float(best_value)}

    return best_params, float(best_value), cv_summary


def optimize_random_forest_hyperparameters(
    parquet_path: Path,
    target_column: str,
    *,
    n_trials: int = HYPEROPT_DEFAULT_N_TRIALS,
    random_state: int = DEFAULT_RANDOM_STATE,
    cv_n_splits: int = CV_N_SPLITS,
) -> tuple[dict[str, Any], float, dict[str, float]]:
    """Run Optuna hyperparameter search for a RandomForest regressor.

    Args:
        parquet_path: Path to the Parquet file with pre-split data (train/test).
        target_column: Name of the target column inside the splits.

    Keyword Args:
        n_trials: Number of Optuna trials to execute.
        random_state: RNG seed used for reproducibility (sampler + model factories).
        cv_n_splits: Number of folds for K-fold cross-validation applied to
            the train set.

    Returns:
        A tuple of (best_params, best_value, cv_summary) where `best_params`
        is the dictionary of hyperparameters found by Optuna, `best_value` is
        the cross-validated MSE (float) minimized by the study, and `cv_summary`
        contains a small mapping of cross-validation metrics (e.g. {"cv_mse": ...}).

    Raises:
        RuntimeError: If the Optuna study completes but no best trial value is
            available (should not happen in normal runs).
    """
    X, y = _build_cv_dataset(parquet_path, target_column)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    kf = KFold(n_splits=cv_n_splits, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "max_depth": trial.suggest_int("max_depth", 4, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 40),
            "max_features": trial.suggest_float("max_features", 0.2, 1.0),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "n_jobs": -1,
            "random_state": random_state,
        }

        mse_values: list[float] = []

        for train_idx, val_idx in kf.split(X):
            # Fold-wise encoding for RF as well
            X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
            (X_train_enc, X_val_enc), _enc = _utils.fit_label_encoders_on_train(
                X_train_raw,
                [X_val_raw],
            )
            X_train = _utils.ensure_numeric_columns(X_train_enc)
            X_val = _utils.ensure_numeric_columns(X_val_enc)
            y_train, y_val = y[train_idx], y[val_idx]

            model = create_random_forest_regressor(random_state=random_state)
            allowed = set(model.get_params().keys())
            safe_params = {k: v for k, v in params.items() if k in allowed}
            model.set_params(**safe_params)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            fold_mse = float(mean_squared_error(np.asarray(y_val), np.asarray(y_pred)))
            mse_values.append(fold_mse)

        return float(np.mean(mse_values))

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_value = study.best_trial.value
    if best_value is None:
        raise RuntimeError("Optuna study completed but best_trial.value is None.")

    best_params: dict[str, Any] = study.best_trial.params
    cv_summary = {"cv_mse": float(best_value), "val_mse": float(best_value)}

    return best_params, float(best_value), cv_summary
