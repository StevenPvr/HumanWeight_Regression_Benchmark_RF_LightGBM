"""Ridge regression baseline model.

This module provides Ridge regression utilities for training and evaluation.
Ridge is used as a linear baseline to compare against more complex models
like Random Forest and LightGBM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.constants import CV_N_SPLITS, DEFAULT_RANDOM_STATE, HYPEROPT_DEFAULT_N_TRIALS
from src.utils import get_logger, load_splits_from_parquet, split_features_target

LOGGER = get_logger(__name__)


def _split_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return categorical and numeric column lists based on dtypes.

    WHY: ColumnTransformer needs explicit column groups. We infer them from
    pandas dtypes to keep behavior simple and consistent.
    """
    cat_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
    num_cols = [c for c in df.columns if c not in cat_cols]
    return cat_cols, num_cols


def _build_ridge_pipeline(alpha: float, *, cat_cols: Sequence[str], num_cols: Sequence[str]) -> Pipeline:
    """Create a preprocessing + Ridge pipeline.

    Steps:
      - OneHotEncode categoricals (drop='first', handle_unknown='ignore')
      - Pass-through numeric columns
      - Standardize all features
      - Fit Ridge with provided alpha
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                list(cat_cols),
            ),
            ("num", "passthrough", list(num_cols)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    return pipe


def optimize_ridge_hyperparameters(
    parquet_path: Path,
    target_column: str,
    *,
    n_trials: int = HYPEROPT_DEFAULT_N_TRIALS,
    random_state: int = DEFAULT_RANDOM_STATE,
    cv_n_splits: int = CV_N_SPLITS,
) -> tuple[dict[str, Any], float, dict[str, float]]:
    """Run Optuna hyperparameter search for a Ridge regressor.

    WHY: Ridge regression requires tuning of the regularization parameter alpha
    to balance bias and variance. Cross-validation on the training set ensures
    no leakage to the test set.

    Args:
        parquet_path: Path to the Parquet file with pre-split data (train/test).
        target_column: Name of the target column inside the splits.

    Keyword Args:
        n_trials: Number of Optuna trials to execute.
        random_state: RNG seed used for reproducibility (sampler + model).
        cv_n_splits: Number of folds for K-fold cross-validation applied to
            the train set.

    Returns:
        A tuple of (best_params, best_value, cv_summary) where `best_params`
        is the dictionary of hyperparameters found by Optuna, `best_value` is
        the cross-validated MSE (float) minimized by the study, and `cv_summary`
        contains cross-validation metrics.

    Raises:
        RuntimeError: If the Optuna study completes but no best trial value is
            available (should not happen in normal runs).
    """
    # Load raw train split; encoding must be fold-specific to avoid leakage
    parquet_path = Path(parquet_path)
    loaded = load_splits_from_parquet(parquet_path)
    if isinstance(loaded, tuple) and len(loaded) == 3:
        train_df = loaded[0]
    else:
        train_df, _ = loaded  # type: ignore[assignment]

    X_raw, y_series = split_features_target(train_df, target_column)
    y = np.asarray(y_series)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    kf = KFold(n_splits=cv_n_splits, shuffle=True, random_state=random_state)

    def _evaluate_fold(
        X_train_fold: pd.DataFrame,
        X_val_fold: pd.DataFrame,
        y_train_fold: np.ndarray,
        y_val_fold: np.ndarray,
        alpha: float,
    ) -> tuple[float, float, float, float]:
        """Evaluate one fold with proper in-fold preprocessing.

        Returns tuple: (mse, rmse, mae, r2)
        """
        cat_cols, num_cols = _split_feature_columns(X_train_fold)
        model = _build_ridge_pipeline(alpha, cat_cols=cat_cols, num_cols=num_cols)
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        mse = float(mean_squared_error(y_val_fold, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_val_fold, y_pred))
        r2 = float(r2_score(y_val_fold, y_pred))
        return mse, rmse, mae, r2

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 0.01, 100.0, log=True)

        mse_values: list[float] = []
        rmse_values: list[float] = []
        mae_values: list[float] = []
        r2_values: list[float] = []

        for train_idx, val_idx in kf.split(X_raw):
            X_train_fold = X_raw.iloc[train_idx]
            X_val_fold = X_raw.iloc[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

            fold_mse, fold_rmse, fold_mae, fold_r2 = _evaluate_fold(
                X_train_fold, X_val_fold, y_train_fold, y_val_fold, alpha
            )
            mse_values.append(fold_mse)
            rmse_values.append(fold_rmse)
            mae_values.append(fold_mae)
            r2_values.append(fold_r2)

        # Optuna optimizes MSE mean
        trial.set_user_attr("cv_rmse_mean", float(np.mean(rmse_values)))
        trial.set_user_attr("cv_mae_mean", float(np.mean(mae_values)))
        trial.set_user_attr("cv_r2_mean", float(np.mean(r2_values)))
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
    # Enrich CV summary with multiple metrics
    # Re-run CV once for summary (cheap given small folds), or use attributes saved on best_trial
    # Here we report only the best value and user attrs for simplicity
    cv_summary = {
        "cv_mse_mean": float(best_value),
        "cv_rmse_mean": float(study.best_trial.user_attrs.get("cv_rmse_mean", float("nan"))),
        "cv_mae_mean": float(study.best_trial.user_attrs.get("cv_mae_mean", float("nan"))),
        "cv_r2_mean": float(study.best_trial.user_attrs.get("cv_r2_mean", float("nan"))),
        # Backward compatibility
        "cv_mse": float(best_value),
    }

    return best_params, float(best_value), cv_summary


def _load_train_data(
    parquet_path: Path,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load raw training data (no encoding/scaling) from parquet file.

    Args:
        parquet_path: Path to the concatenated Parquet file with splits.
        target_column: Name of the target column.

    Returns:
        Tuple of (X_train, y_train) where X_train is numeric features.
    """
    parquet_path = Path(parquet_path)
    LOGGER.info("Loading dataset splits from %s", parquet_path)

    loaded = load_splits_from_parquet(parquet_path)
    if isinstance(loaded, tuple) and len(loaded) == 3:
        train_df = loaded[0]
    else:
        train_df, _ = loaded  # type: ignore[assignment]

    X_train, y_train = split_features_target(train_df, target_column)

    LOGGER.info("Train split: %d rows", len(y_train))
    LOGGER.info("Train shape: %s", X_train.shape)

    return X_train, y_train


# Removed legacy standardization helpers; the Pipeline encapsulates preprocessing.


class _IdentityScaler:
    """Simple passthrough transformer to keep CLI contract for scaler artifact.

    WHY: After migrating to a unified sklearn Pipeline, an external scaler is
    no longer needed. We persist a no-op transformer so existing evaluation
    code can call ``transform`` without changes.
    """

    def transform(self, X: Any) -> Any:  # noqa: D401 - trivial pass-through
        return X


def train_ridge_with_best_params(
    parquet_path: Path,
    params_json_path: Path,
    *,
    target_column: str,
    random_state: int = DEFAULT_RANDOM_STATE,

) -> tuple[Pipeline, _IdentityScaler]:
    """Train a Ridge pipeline (OHE + StandardScaler + Ridge) with best params.

    WHY: Ridge regression is sensitive to feature scale. StandardScaler ensures
    all features are centered (mean=0) and scaled (std=1) before training.
    The scaler is fit only on training data to prevent test leakage.

    Args:
        parquet_path: Path to the concatenated Parquet file that contains the
            train/test splits with encoded categorical columns.
        params_json_path: Path to the JSON file containing best hyperparameters
            from Optuna optimization.
        target_column: Name of the target column in the dataset.
        random_state: Random seed for reproducibility.

    Returns:
        A tuple of (trained_model, fitted_scaler) where the scaler was fit
        on training data and used to transform both train and test features.

    Raises:
        FileNotFoundError: If parquet_path or params_json_path do not exist.
        KeyError: If the JSON file does not contain expected 'best_params' key.
    """
    from src.utils import read_best_params

    X_train, y_train = _load_train_data(parquet_path, target_column)

    params_json_path = Path(params_json_path)
    LOGGER.info("Loading best hyperparameters from %s", params_json_path)
    best_params = read_best_params(params_json_path)
    LOGGER.info("Best hyperparameters: %s", best_params)

    alpha = best_params.get("alpha", 1.0)
    cat_cols, num_cols = _split_feature_columns(X_train)
    model = _build_ridge_pipeline(alpha, cat_cols=cat_cols, num_cols=num_cols)
    LOGGER.info("Fitting Ridge pipeline with alpha=%.4f", alpha)
    model.fit(X_train, y_train)

    return model, _IdentityScaler()
