from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb  # type: ignore
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error

from src.constants import (
    DEFAULT_RANDOM_STATE,
    HYPEROPT_DEFAULT_N_TRIALS,
    HYPEROPT_LGBM_EARLY_STOPPING_ROUNDS,
)
from src.models.models import (
    create_lightgbm_regressor,
    create_random_forest_regressor,
)
from src.utils import prepare_train_val_numeric_splits


def optimize_lightgbm_hyperparameters(
    parquet_path: Path,
    target_column: str,
    *,
    n_trials: int = HYPEROPT_DEFAULT_N_TRIALS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[dict[str, Any], float, dict[str, float]]:
    """
    Optimize LightGBM hyperparameters using a single train/validation split.

    Args:
        parquet_path: Path to the concatenated Parquet file with a 'split' column.
        target_column: Name of the target column present in all splits.
        n_trials: Number of Optuna trials to run (default: 100).
        random_state: Seed for reproducibility across sampler and model.

    Returns:
        Tuple (best_params, best_value, val_summary) where:
            - best_params: dict of best hyperparameters found by Optuna.
            - best_value: minimal validation MSE achieved.
            - val_summary: {"val_mse": best_value} for convenience.
    """
    parquet_path = Path(parquet_path)
    train_data, val_data = prepare_train_val_numeric_splits(parquet_path, target_column)
    train_features, y_train = train_data
    val_features, y_val = val_data
    # WHY: compute encoded features once so every Optuna trial reuses the same holdout split.

    # Reduce Optuna logger noise to only warnings and errors.
    # WHY: HPO runs are iterative; excessive INFO logs clutter the console.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Silence LightGBM's internal logger during optimization to avoid noisy C++ logs.
    # WHY: Messages like "No further splits with positive gain" are not actionable here.
    class _SilentLogger:
        """Minimal logger that drops LightGBM info/warning messages.

        WHY: Keep console output focused on Optuna results, not training internals.
        """

        def info(self, msg: str) -> None:  # pragma: no cover - trivial sink
            return None

        def warning(self, msg: str) -> None:  # pragma: no cover - trivial sink
            return None

    lgb.register_logger(_SilentLogger())

    def objective(trial: optuna.Trial) -> float:
        # WHY: Keep search simple yet elastic enough for high-trial Optuna runs
        params: dict[str, Any] = {
            # WHY: 100-trial budget merits wider ensembles to surface high-capacity fits
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 512),
            # WHY: allow deeper trees yet keep small depths available for regularisation
            "max_depth": trial.suggest_int("max_depth", 3, 18),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 400),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            # WHY: Looser regularisation bounds unlock sparse vs dense solutions per feature
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 80.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 40.0),
            "n_jobs": -1,
            "random_state": random_state,
            # WHY: silence LightGBM training logs and warnings during HPO runs
            "verbosity": -1,
        }

        model = create_lightgbm_regressor(random_state=random_state)
        # Apply only supported params (robust to version differences)
        allowed = set(model.get_params().keys())
        safe_params = {k: v for k, v in params.items() if k in allowed}
        model.set_params(**safe_params)

        # WHY: calm training output and remove periodic metric logs
        callbacks = [
            # Avoid early-stopping status lines ("Training until...", "Best iteration is:")
            lgb.early_stopping(
                stopping_rounds=HYPEROPT_LGBM_EARLY_STOPPING_ROUNDS,
                verbose=False,
            ),
            # Disable periodic metric printing
            lgb.log_evaluation(period=0),
        ]
        model.fit(
            train_features,
            y_train,
            eval_set=[(val_features, y_val)],
            eval_metric="l2",
            callbacks=callbacks,
        )
        best_iter = getattr(model, "best_iteration_", None)
        y_pred = model.predict(val_features, num_iteration=best_iter)
        mse = float(mean_squared_error(np.asarray(y_val), np.asarray(y_pred)))
        return mse

    study = optuna.create_study(
        direction="minimize", sampler=TPESampler(seed=random_state)
    )
    # Show Optuna progress bar by default while keeping logs minimal.
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_value = study.best_trial.value
    if best_value is None:
        raise RuntimeError("Optuna study completed but best_trial.value is None.")

    best_params: dict[str, Any] = study.best_trial.params
    val_summary = {"val_mse": float(best_value)}
    return best_params, float(best_value), val_summary


def optimize_random_forest_hyperparameters(
    parquet_path: Path,
    target_column: str,
    *,
    n_trials: int = HYPEROPT_DEFAULT_N_TRIALS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[dict[str, Any], float, dict[str, float]]:
    """
    Optimize RandomForest hyperparameters using a single train/validation split.

    WHY: Add a complementary baseline after LightGBM while staying simple. We
    reuse the same fold generation, numeric encoding, and scoring as LightGBM
    for consistency and comparability across models.

    Args:
        parquet_path: Path to the concatenated Parquet file with a 'split' column.
        target_column: Name of the target column present in all splits.
        n_trials: Number of Optuna trials to run (default: 100).
        random_state: Seed for reproducibility across sampler and model.

    Returns:
        Tuple (best_params, best_value, val_summary).

        WHY: Keep `val_summary` as `Dict[str, float]` (same as LightGBM) to
        avoid invariant `Dict` variance issues flagged by type checkers and to
        maintain a simple, consistent contract.
    """
    parquet_path = Path(parquet_path)
    train_data, val_data = prepare_train_val_numeric_splits(parquet_path, target_column)
    train_features, y_train = train_data
    val_features, y_val = val_data
    # WHY: preprocessing here mirrors the LightGBM path so comparisons stay fair.

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        # WHY: Mirror LightGBM expansion so forests can scale with the larger trial budget
        params: dict[str, Any] = {
            # WHY: Expand ranges so 100 Optuna trials can explore heavier ensembles too
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "max_depth": trial.suggest_int("max_depth", 4, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 40),
            "max_features": trial.suggest_float(
                "max_features", 0.2, 1.0
            ),  # fraction of features
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "n_jobs": -1,
            "random_state": random_state,
        }
        model = create_random_forest_regressor(random_state=random_state)
        allowed = set(model.get_params().keys())
        safe_params = {k: v for k, v in params.items() if k in allowed}
        model.set_params(**safe_params)

        model.fit(train_features, y_train)
        y_pred = model.predict(val_features)
        mse = float(mean_squared_error(np.asarray(y_val), np.asarray(y_pred)))
        return mse

    study = optuna.create_study(
        direction="minimize", sampler=TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_value = study.best_trial.value
    if best_value is None:
        raise RuntimeError("Optuna study completed but best_trial.value is None.")

    best_params: dict[str, Any] = study.best_trial.params
    val_summary = {"val_mse": float(best_value)}

    return best_params, float(best_value), val_summary
