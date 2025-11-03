"""CLI to train and save gradient boosting and random forest models.

Usage example:
    python -m src.training.main \
        --parquet data/processed/splits.parquet \
        --params results/best_lightgbm_params.json \
        --out results/models/lightgbm

This will train using the best params and save to
`results/models/lightgbm.joblib` by default.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
# WHY: allow running the module as a script without installing the package by ensuring imports resolve
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.constants import (
    DEFAULT_LIGHTGBM_MODEL_PATH,
    DEFAULT_LIGHTGBM_PARAMS_PATH,
    DEFAULT_RANDOM_FOREST_MODEL_PATH,
    DEFAULT_RANDOM_FOREST_PARAMS_PATH,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TRAINING_PARQUET_PATH,
    PROJECT_ROOT,
    TARGET_COLUMN,
)

project_root = PROJECT_ROOT
DEFAULT_PARQUET = DEFAULT_TRAINING_PARQUET_PATH
DEFAULT_PARAMS = DEFAULT_LIGHTGBM_PARAMS_PATH
DEFAULT_MODEL_OUT = DEFAULT_LIGHTGBM_MODEL_PATH
DEFAULT_RF_PARAMS = DEFAULT_RANDOM_FOREST_PARAMS_PATH
DEFAULT_RF_MODEL_OUT = DEFAULT_RANDOM_FOREST_MODEL_PATH
# WHY: allow running the module as a script without installing the package by ensuring imports resolve
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training.training import (
    train_lightgbm_with_best_params,
    save_lightgbm_model,
)
from src.models.models import create_random_forest_regressor
from src.utils import (
    load_splits_from_parquet,
    read_best_params,
    save_training_results,
    split_features_target,
    ensure_numeric_columns,
    get_logger,
    to_project_relative_path,
)


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class ModelStageResult:
    """Container for model artifact path, fitted estimator, and diagnostics."""

    artifact_path: str | None = None
    model: object | None = None
    validation_mse: float | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    """Configure CLI arguments for the training entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Train and save a LightGBM regressor using best hyperparameters. "
            "Optionally also train and save a RandomForest right after."
        ),
    )
    parser.add_argument(
        "--parquet",
        default=str(DEFAULT_PARQUET),
        help=(
            "Path to concatenated Parquet with 'split' column (train/val/test). "
            f"Default: {DEFAULT_PARQUET}"
        ),
    )
    parser.add_argument(
        "--params",
        default=str(DEFAULT_PARAMS),
        help=(
            "Path to JSON containing {'best_params': {...}} (LightGBM). "
            f"Default: {DEFAULT_PARAMS}"
        ),
    )
    parser.add_argument(
        "--rf-params",
        default=str(DEFAULT_RF_PARAMS),
        help=(
            "Path to JSON containing {'best_params': {...}} (RandomForest). "
            f"Default: {DEFAULT_RF_PARAMS}"
        ),
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_MODEL_OUT),
        help=(
            "Output model path (extension optional, defaults to .joblib). "
            f"Default directory: {DEFAULT_MODEL_OUT}"
        ),
    )
    parser.add_argument(
        "--rf-out",
        default=str(DEFAULT_RF_MODEL_OUT),
        help=(
            "Output RandomForest model path (extension optional). "
            f"Default directory: {DEFAULT_RF_MODEL_OUT}"
        ),
    )
    parser.add_argument(
        "--target-column",
        default=TARGET_COLUMN,
        help=f"Target column name (default: {TARGET_COLUMN})",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random state for reproducibility (default: {DEFAULT_RANDOM_STATE})",
    )
    # Model selection: new opt-in flag while keeping backward-compatible toggle
    parser.add_argument(
        "--models",
        choices=["lightgbm", "random_forest", "both"],
        default=None,  # None → fallback to legacy --also-random-forest behavior
        help=(
            "Which model(s) to train: 'lightgbm', 'random_forest', or 'both'. "
            "If omitted, the legacy --also-random-forest flag controls behavior."
        ),
    )
    parser.add_argument(
        "--also-random-forest",
        action="store_true",
        dest="also_random_forest",
        help="After LightGBM, also train and save a RandomForest.",
    )
    parser.add_argument(
        "--no-random-forest",
        action="store_false",
        dest="also_random_forest",
        help="Skip the RandomForest training phase.",
    )
    parser.set_defaults(also_random_forest=True)
    return parser


def determine_model_selection(args: argparse.Namespace) -> str:
    """Return which models to train based on CLI flags and optional prompt."""

    if args.models is not None:
        return str(args.models)

    try:
        if sys.stdin.isatty():
            choice = input(
                "Select models to train: [1] LightGBM, [2] RandomForest, [3] Both (default 3): "
            ).strip()
            if choice == "1":
                return "lightgbm"
            if choice == "2":
                return "random_forest"
    except Exception:  # pragma: no cover - interactive safeguards
        LOGGER.debug("Falling back to legacy selection because interactive prompt failed")

    return "both" if bool(args.also_random_forest) else "lightgbm"


def load_dataset_splits_safe(parquet_path: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Load dataset splits while logging any failure without aborting the CLI."""

    try:
        train_df, val_df, test_df = load_splits_from_parquet(parquet_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to load dataset splits: %s", exc)
        return None, None, None

    LOGGER.info(
        "Dataset splits loaded for diagnostics (train=%d, val=%d)",
        len(train_df),
        len(val_df),
    )
    return train_df, val_df, test_df


def save_lightgbm_summary(
    args: argparse.Namespace,
    artifact_path: str,
    train_df: pd.DataFrame | None,
    val_df: pd.DataFrame | None,
    validation_mse: float | None,
    best_params: dict[str, object] | None,
) -> None:
    """Persist LightGBM training summary next to the artifact."""

    model_path = Path(artifact_path)
    use_default_out = Path(args.out) == DEFAULT_MODEL_OUT
    base_name = "lightgbm" if use_default_out else model_path.stem
    metrics_path = model_path.parent / f"{base_name}_metrics.json"

    summary_payload: dict[str, object] = {
        "model_path": str(to_project_relative_path(artifact_path)),
        "parquet_path": str(to_project_relative_path(args.parquet)),
        "params_path": str(to_project_relative_path(args.params)),
        "target_column": args.target_column,
        "random_state": args.random_state,
        "validation": {"mse": validation_mse},
    }
    if train_df is not None:
        summary_payload["training_rows"] = len(train_df)
    if val_df is not None:
        summary_payload["validation_rows"] = len(val_df)
    if best_params is not None:
        summary_payload["best_params"] = best_params

    save_training_results(summary_payload, str(metrics_path))
    LOGGER.info(
        "Training summary saved to %s",
        to_project_relative_path(metrics_path),
    )


def save_random_forest_summary(
    args: argparse.Namespace,
    artifact_path: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    validation_mse: float,
    best_params: dict[str, object],
) -> None:
    """Store RandomForest training diagnostics alongside the model artifact."""

    rf_base = Path(artifact_path).stem
    rf_metrics_path = Path(artifact_path).parent / f"{rf_base}_metrics.json"
    rf_summary: dict[str, object] = {
        "model_path": str(to_project_relative_path(artifact_path)),
        "parquet_path": str(to_project_relative_path(args.parquet)),
        "params_path": str(to_project_relative_path(args.rf_params)),
        "target_column": args.target_column,
        "random_state": args.random_state,
        "validation": {"mse": validation_mse},
        "training_rows": len(train_df),
        "validation_rows": len(val_df),
        "best_params": best_params,
    }
    save_training_results(rf_summary, str(rf_metrics_path))
    LOGGER.info(
        "RandomForest training summary saved to %s",
        to_project_relative_path(rf_metrics_path),
    )


def run_lightgbm_stage(
    args: argparse.Namespace,
    selected: str,
    train_df: pd.DataFrame | None,
    val_df: pd.DataFrame | None,
) -> ModelStageResult:
    """Train LightGBM when requested, persist artifact, and log diagnostics."""

    if selected not in {"lightgbm", "both"}:
        return ModelStageResult()

    model = train_lightgbm_with_best_params(
        parquet_path=args.parquet,
        params_json_path=args.params,
        target_column=args.target_column,
        random_state=args.random_state,
    )
    LOGGER.info("Model trained; persisting artifact")
    artifact_path = save_lightgbm_model(model, args.out)
    LOGGER.info("Model saved to %s", artifact_path)

    validation_mse: float | None = None
    if val_df is not None:
        try:
            X_val_raw, y_val = split_features_target(val_df, args.target_column)
            X_val = ensure_numeric_columns(X_val_raw)
            num_iter = getattr(model, "best_iteration_", None)
            try:
                preds = model.predict(X_val, num_iteration=num_iter)
            except TypeError:
                preds = model.predict(X_val)
            validation_mse = float(mean_squared_error(np.asarray(y_val), np.asarray(preds)))
            LOGGER.info("Validation MSE (final model): %.6f", validation_mse)
        except Exception as exc:  # pragma: no cover - diagnostics only
            LOGGER.warning("Failed computing validation MSE: %s", exc)

    best_params = None
    with suppress(Exception):
        best_params = read_best_params(args.params)

    save_lightgbm_summary(args, artifact_path, train_df, val_df, validation_mse, best_params)

    return ModelStageResult(artifact_path=artifact_path, model=model, validation_mse=validation_mse)


def run_random_forest_stage(
    args: argparse.Namespace,
    selected: str,
    train_df: pd.DataFrame | None,
    val_df: pd.DataFrame | None,
) -> ModelStageResult:
    """Optionally train a RandomForest after LightGBM and record diagnostics."""

    if selected not in {"random_forest", "both"}:
        return ModelStageResult()
    if train_df is None or val_df is None:
        LOGGER.warning("Skipping RandomForest stage because dataset splits are unavailable")
        return ModelStageResult()

    rf_best = read_best_params(args.rf_params)
    rf = create_random_forest_regressor(random_state=int(args.random_state))
    allowed = set(rf.get_params().keys())
    rf_safe = {k: v for k, v in rf_best.items() if k in allowed}
    # WHY: ensure parallel execution even if tuning omitted n_jobs (mirrors LightGBM)
    rf_safe.setdefault("n_jobs", -1)
    rf.set_params(**rf_safe)
    LOGGER.info(
        "RandomForest hyperparameters after merge: n_estimators=%s | n_jobs=%s",
        rf.get_params().get("n_estimators"),
        rf.get_params().get("n_jobs"),
    )

    X_train_raw, y_train = split_features_target(train_df, args.target_column)
    X_val_raw, y_val = split_features_target(val_df, args.target_column)
    X_train = ensure_numeric_columns(X_train_raw)
    X_val = ensure_numeric_columns(X_val_raw)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_val)
    validation_mse = float(mean_squared_error(np.asarray(y_val), np.asarray(preds)))
    LOGGER.info("RandomForest Validation MSE: %.6f", validation_mse)

    rf_out_path = str(args.rf_out)
    if not rf_out_path.lower().endswith((".joblib", ".pkl")):
        rf_out_path = f"{rf_out_path}.joblib"
    Path(rf_out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib_module = sys.modules.get("joblib", joblib)
    joblib_module.dump(rf, rf_out_path)
    LOGGER.info("RandomForest saved to %s", rf_out_path)

    save_random_forest_summary(args, rf_out_path, train_df, val_df, validation_mse, rf_best)

    return ModelStageResult(artifact_path=rf_out_path, model=rf, validation_mse=validation_mse)


def main(argv: list[str] | None = None) -> int:
    """Execute CLI workflow: train model, report metrics, persist artifact."""
    parser = build_arg_parser()
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv_list)

    LOGGER.info("Starting training pipeline")
    LOGGER.info("Using dataset: %s", args.parquet)
    LOGGER.info("Using parameters: %s", args.params)
    LOGGER.info("Model artifact target: %s", args.out)
    LOGGER.info("Target column: %s | Random state: %d", args.target_column, args.random_state)

    selected = determine_model_selection(args)
    train_df, val_df, _ = load_dataset_splits_safe(args.parquet)

    lightgbm_result = run_lightgbm_stage(args, selected, train_df, val_df)
    rf_result = run_random_forest_stage(args, selected, train_df, val_df)

    final_path = lightgbm_result.artifact_path or rf_result.artifact_path
    if final_path is None:
        LOGGER.error("No model was trained. Check --models selection.")
        return 1

    LOGGER.info("Trained model artifact saved to %s", final_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
