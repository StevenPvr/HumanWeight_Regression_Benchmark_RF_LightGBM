"""Hyperparameters optimization pipeline entrypoint.

Runs the pipeline:
  1) Optionally infer the target column from common variants
  2) Run Optuna optimization using a single train/val split (no CV)
  3) Persist best hyperparameters and validation MSE to results JSON

Defaults point to repository paths but can be overridden via CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
# WHY: allow running the module as a script without installing the package by ensuring imports resolve
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse  # noqa: E402
from typing import Optional, Sequence  # noqa: E402

from src.constants import (  # noqa: E402
    DEFAULT_HYPEROPT_INPUT_PARQUET,
    DEFAULT_RANDOM_STATE,
    HYPEROPT_DEFAULT_N_TRIALS,
    HYPEROPT_LGBM_RESULTS_FILENAME,
    HYPEROPT_RF_RESULTS_FILENAME,
    HYPEROPT_TARGET_COLUMN_CANDIDATES,
    RESULTS_DIR,
)
from src.hyperparameters_optimization.optimization import (  # noqa: E402
    optimize_lightgbm_hyperparameters,
    optimize_random_forest_hyperparameters,
)
from src.utils import get_logger, load_splits_from_parquet, save_training_results  # noqa: E402

LOGGER = get_logger(__name__)


def _infer_target_column(columns: Sequence[str]) -> str:
    """Infer the target column name from known variants.

    Args:
        columns: Column names available in the dataset.

    Returns:
        The detected target column name.

    Raises:
        ValueError: If no known target column variant is present.
    """
    for name in HYPEROPT_TARGET_COLUMN_CANDIDATES:
        if name in columns:
            return name
    raise ValueError(
        "Could not infer target column. Provide --target explicitly. "
        f"Available columns: {list(columns)[:30]}..."
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for this module.

    WHY: Keep ``main`` focused on orchestration and under line limits.
    """
    parser = argparse.ArgumentParser(description="Run hyperparameters optimization pipeline")
    parser.add_argument(
        "--input",
        "-i",
        default=str(DEFAULT_HYPEROPT_INPUT_PARQUET),
        help="Path to concatenated pre-split Parquet with 'split' column (encoded)",
    )
    parser.add_argument(
        "--target",
        "-t",
        default=None,
        help="Target column name (if omitted, inferred among common variants)",
    )
    parser.add_argument(
        "--n-trials",
        "-n",
        type=int,
        default=HYPEROPT_DEFAULT_N_TRIALS,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for sampler and model",
    )
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Which model(s) to optimize: 'lightgbm' (alias: lgbm), "
            "'random_forest' (aliases: randomforest, random-forest, rf), or 'both'. "
            "If omitted in an interactive terminal, you will be prompted."
        ),
    )
    return parser


def _normalize_models_choice(models: Optional[str]) -> Optional[str]:
    """Return a canonical model selection string from aliases.

    Accepts common variants like 'lgbm', 'rf', 'random-forest' and maps them to
    'lightgbm' or 'random_forest'. Unknown values become None for interactive fallback.
    """
    if not isinstance(models, str):
        return None
    norm = models.strip().lower().replace(" ", "-")
    if norm in {"lgbm"}:
        return "lightgbm"
    if norm in {"randomforest", "random-forest", "rf"}:
        return "random_forest"
    if norm in {"lightgbm", "random_forest", "both"}:
        return norm
    return None


def _resolve_models_choice(models: Optional[str]) -> str:
    """Resolve the model selection, prompting the user only in TTY.

    WHY: Keep non-interactive runs deterministic (default to 'both') while still
    offering a prompt for manual invocations.
    """
    normalized = _normalize_models_choice(models)
    if normalized is not None:
        return normalized
    try:
        if sys.stdin.isatty():
            choice = input(
                "Select models to optimize: [1] LightGBM, [2] RandomForest, [3] Both (default 3): "
            ).strip()
            if choice == "1":
                return "lightgbm"
            if choice == "2":
                return "random_forest"
            return "both"
        return "both"
    except Exception:
        return "both"


def _infer_target_from_data(input_path: Path, explicit: Optional[str]) -> str:
    """Return the target column, inferring from data when not provided.

    Args:
        input_path: Path to the encoded splits Parquet file.
        explicit: Optional target name from CLI.
    """
    if explicit is not None:
        return explicit
    splits = load_splits_from_parquet(input_path)
    train_df = splits[0]
    return _infer_target_column(list(train_df.columns))


def _compute_data_summary(input_path: Path) -> dict[str, int]:
    """Return a minimal data summary for result payloads."""
    splits = load_splits_from_parquet(input_path)
    tv_train = splits[0]
    return {"train_rows": int(len(tv_train))}


def _run_lightgbm(
    *,
    parquet_path: Path,
    target_column: str,
    n_trials: int,
    random_state: int,
    data_summary: dict[str, int],
) -> None:
    """Execute LightGBM optimization branch and persist results."""
    best_params, best_value, _ = optimize_lightgbm_hyperparameters(
        parquet_path=parquet_path,
        target_column=target_column,
        n_trials=n_trials,
        random_state=random_state,
    )
    output_path = RESULTS_DIR / HYPEROPT_LGBM_RESULTS_FILENAME
    saved = save_training_results(
        {
            "best_params": best_params,
            "best_value": float(best_value),
            "data_summary": data_summary,
        },
        output_path,
    )
    LOGGER.info("[Hyperopt][LGBM] Best hyperparameters:")
    for k, v in best_params.items():
        LOGGER.info("  - %s: %s", k, v)
    LOGGER.info("[Hyperopt][LGBM] Results saved to: %s", saved)


def _run_random_forest(
    *,
    parquet_path: Path,
    target_column: str,
    n_trials: int,
    random_state: int,
    data_summary: dict[str, int],
) -> None:
    """Execute RandomForest optimization branch and persist results."""
    LOGGER.info("[Hyperopt] Optimizing RandomForest...")
    best_params, best_val, _ = optimize_random_forest_hyperparameters(
        parquet_path=parquet_path,
        target_column=target_column,
        n_trials=n_trials,
        random_state=random_state,
    )
    rf_out = RESULTS_DIR / HYPEROPT_RF_RESULTS_FILENAME
    saved = save_training_results(
        {
            "best_params": best_params,
            "best_value": float(best_val),
            "data_summary": data_summary,
        },
        rf_out,
    )
    LOGGER.info("[Hyperopt][RF] Best hyperparameters:")
    for k, v in best_params.items():
        LOGGER.info("  - %s: %s", k, v)
    LOGGER.info("[Hyperopt][RF] Results saved to: %s", saved)


def main() -> None:
    """CLI entrypoint wrapper delegating to smaller helpers."""
    parser = _build_arg_parser()
    args = parser.parse_args(list(sys.argv[1:]))

    input_path = Path(args.input)
    models = _resolve_models_choice(args.models)
    target_col = _infer_target_from_data(input_path, args.target)
    data_summary = _compute_data_summary(input_path)

    if models in ("lightgbm", "both"):
        _run_lightgbm(
            parquet_path=input_path,
            target_column=target_col,
            n_trials=args.n_trials,
            random_state=args.random_state,
            data_summary=data_summary,
        )

    if models in ("random_forest", "both"):
        _run_random_forest(
            parquet_path=input_path,
            target_column=target_col,
            n_trials=args.n_trials,
            random_state=args.random_state,
            data_summary=data_summary,
        )


if __name__ == "__main__":
    main()
