"""Run the full pipeline sequentially with explicit stage ordering."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

project_root = Path(__file__).resolve().parent.parent
# WHY: allow running as a script without requiring package installation
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.baseline.main import main as baseline_main  # noqa: E402
from src.constants import (
    DEFAULT_RIDGE_MODEL_PATH,
    DEFAULT_RIDGE_PARAMS_PATH,
    DEFAULT_RIDGE_SCALER_PATH,
    DEFAULT_TRAINING_PARQUET_PATH,
    TARGET_COLUMN,
)
from src.data_cleaning.main import main as data_cleaning_main  # noqa: E402
from src.data_preparation.main import main as data_preparation_main  # noqa: E402
from src.eval.main import main as eval_main  # noqa: E402
from src.feature_engineering.main import main as feature_engineering_main  # noqa: E402
from src.hyperparameters_optimization.main import main as hyperparameters_main  # noqa: E402
from src.training.main import main as training_main  # noqa: E402
from src.utils import get_logger  # noqa: E402

LOGGER = get_logger(__name__)


def main() -> None:
    """Run the project pipeline end-to-end using stage-main wrappers."""
    stages: list[tuple[str, Callable[..., object]]] = [
        ("data_cleaning", data_cleaning_main),
        ("data_preparation", data_preparation_main),
        ("hyperparameters_optimization", hyperparameters_main),
        ("feature_engineering", feature_engineering_main),
        ("training", training_main),
    ]
    # WHY: explicit list preserves deterministic order across future refactors
    for label, runner in stages:
        LOGGER.info("Running %s", label)
        runner()

    # Run Ridge baseline (optimize + train) on produced splits with defaults
    # WHY: Ensure baseline artifacts are reproducible from the global runner
    LOGGER.info("Running baseline: ridge optimize")
    baseline_main(
        [
            "--parquet",
            str(DEFAULT_TRAINING_PARQUET_PATH),
            "--target-column",
            TARGET_COLUMN,
            "--optimize",
            "--params-out",
            str(DEFAULT_RIDGE_PARAMS_PATH),
        ]
    )
    LOGGER.info("Running baseline: ridge train")
    baseline_main(
        [
            "--parquet",
            str(DEFAULT_TRAINING_PARQUET_PATH),
            "--target-column",
            TARGET_COLUMN,
            "--train",
            "--params",
            str(DEFAULT_RIDGE_PARAMS_PATH),
            "--model-out",
            str(DEFAULT_RIDGE_MODEL_PATH),
            "--scaler-out",
            str(DEFAULT_RIDGE_SCALER_PATH),
        ]
    )

    # Run evaluation on all models (including Ridge baseline)
    # WHY: Evaluate all trained models after they are available
    LOGGER.info("Running eval")
    eval_main()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
