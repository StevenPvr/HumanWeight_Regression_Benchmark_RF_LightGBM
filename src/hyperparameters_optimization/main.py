"""Hyperparameters optimization pipeline entrypoint.

Runs the pipeline:
  1) Optionally infer the target column from common variants
  2) Run Optuna optimization using a single train/val split (no CV)
  3) Persist best hyperparameters and validation MSE to results JSON

Defaults point to repository paths but can be overridden via CLI.
"""

from __future__ import annotations

from pathlib import Path
import sys
project_root = Path(__file__).parent.parent.parent
# WHY: allow running the module as a script without installing the package by ensuring imports resolve
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
from typing import Sequence

from src.constants import (
    DEFAULT_HYPEROPT_INPUT_PARQUET,
    DEFAULT_RANDOM_STATE,
    HYPEROPT_DEFAULT_N_TRIALS,
    HYPEROPT_LGBM_RESULTS_FILENAME,
    HYPEROPT_RF_RESULTS_FILENAME,
    HYPEROPT_TARGET_COLUMN_CANDIDATES,
    RESULTS_DIR,
)
from src.utils import get_logger, load_splits_from_parquet, save_training_results
from src.hyperparameters_optimization.optimization import (
    optimize_lightgbm_hyperparameters,
    optimize_random_forest_hyperparameters,
)


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


def main() -> None:
    """Execute the hyperparameter optimization CLI entrypoint.

    WHY: Provide a convenient way to trigger the search without importing the module.

    Returns:
        None.
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

    argv_list = list(sys.argv[1:])
    args = parser.parse_args(argv_list)

    input_path = Path(args.input)

    # Normalize model selection to a canonical value to avoid surprises.
    # WHY: Users often type 'randomforest' or 'rf'; we accept common aliases
    # and map them to 'random_forest' to ensure only the intended branch runs.
    if isinstance(args.models, str):
        norm = args.models.strip().lower().replace(" ", "-")
        if norm in {"lgbm"}:
            args.models = "lightgbm"
        elif norm in {"randomforest", "random-forest", "rf"}:
            args.models = "random_forest"
        elif norm not in {"lightgbm", "random_forest", "both"}:
            # Unknown value: fall back to interactive/default handling below
            args.models = None

    # Infer target name if not provided
    if args.target is None:
        train_df, _val_df, _test_df = load_splits_from_parquet(input_path)
        target_col = _infer_target_column(list(train_df.columns))
    else:
        target_col = args.target

    # Interactive selection when --models not provided and running in a TTY
    if args.models is None:
        try:
            if sys.stdin.isatty():
                choice = input(
                    "Select models to optimize: [1] LightGBM, [2] RandomForest, [3] Both (default 3): "
                ).strip()
                if choice == "1":
                    args.models = "lightgbm"
                elif choice == "2":
                    args.models = "random_forest"
                else:
                    args.models = "both"
            else:
                args.models = "both"
        except Exception:
            args.models = "both"

    # WHY: Load once to compute data summary for all outputs
    tv_train, tv_val, _ = load_splits_from_parquet(input_path)
    data_summary = {
        "train_rows": int(len(tv_train)),
        "val_rows": int(len(tv_val)),
        "train_val_rows": int(len(tv_train) + len(tv_val)),
    }

    # Run selected model(s)
    if args.models in ("lightgbm", "both"):
        best_params, best_value, val_summary = optimize_lightgbm_hyperparameters(
            parquet_path=input_path,
            target_column=target_col,
            n_trials=args.n_trials,
            random_state=args.random_state,
        )

        output_path = RESULTS_DIR / HYPEROPT_LGBM_RESULTS_FILENAME
        saved_lgbm = save_training_results(
            {
                "best_params": best_params,
                "best_value": float(best_value),
                "val_metrics": {"mse": float(best_value)},
                "data_summary": data_summary,
            },
            output_path,
        )
        LOGGER.info("[Hyperopt][LGBM] Validation MSE: %s", best_value)
        LOGGER.info("[Hyperopt][LGBM] Best hyperparameters:")
        for k, v in best_params.items():
            LOGGER.info("  - %s: %s", k, v)
        LOGGER.info("[Hyperopt][LGBM] Results saved to: %s", saved_lgbm)

    if args.models in ("random_forest", "both"):
        LOGGER.info("[Hyperopt] Optimizing RandomForest...")
        rf_best_params, rf_best_val, _ = optimize_random_forest_hyperparameters(
            parquet_path=input_path,
            target_column=target_col,
            n_trials=args.n_trials,
            random_state=args.random_state,
        )
        rf_out = RESULTS_DIR / HYPEROPT_RF_RESULTS_FILENAME
        saved_rf = save_training_results(
            {
                "best_params": rf_best_params,
                "best_value": float(rf_best_val),
                "val_metrics": {"mse": float(rf_best_val)},
                "data_summary": data_summary,
            },
            rf_out,
        )
        LOGGER.info("[Hyperopt][RF] Validation MSE: %s", rf_best_val)
        LOGGER.info("[Hyperopt][RF] Best hyperparameters:")
        for k, v in rf_best_params.items():
            LOGGER.info("  - %s: %s", k, v)
        LOGGER.info("[Hyperopt][RF] Results saved to: %s", saved_rf)


if __name__ == "__main__":
    main()
