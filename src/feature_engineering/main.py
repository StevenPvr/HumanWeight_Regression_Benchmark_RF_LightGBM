"""Feature engineering pipeline entrypoint.

Runs the full pipeline:
  1) Load train/val/test splits from a single Parquet with a 'split' column
  2) Separate features/target
  3) Compute RandomForest permutation feature importance on validation split
  4) Save results to JSON and plot to image

Defaults point to the repository's data/results paths but can be overridden via CLI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.constants import (  # noqa: E402
    DATA_DIR,
    DEFAULT_FEATURE_IMPORTANCE_JSON_PATH,
    DEFAULT_FEATURE_IMPORTANCE_PLOT_PATH,
    DEFAULT_PERMUTATION_N_JOBS,
    DEFAULT_PERMUTATION_REPEATS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SPLITS_PARQUET_FILE,
    TARGET_COLUMN_CANDIDATES,
)
from src.data_preparation.data_preparation import split_train_test  # noqa: E402
from src.feature_engineering.feature_engineering import (  # noqa: E402
    compute_random_forest_permutation_importance,
    plot_feature_importances,
    save_feature_importances_to_json,
)
from src.utils import get_logger, load_splits_from_parquet, split_features_target  # noqa: E402

LOGGER = get_logger(__name__)


def _infer_target_column(columns: Sequence[str]) -> str:
    """Heuristically infer the target column name from known variants."""
    for name in TARGET_COLUMN_CANDIDATES:
        if name in columns:
            return name
    raise ValueError(
        "Could not infer target column. Provide --target explicitly. "
        f"Available columns: {list(columns)[:30]}..."
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for feature engineering."""
    parser = argparse.ArgumentParser(description="Run feature engineering pipeline")
    parser.add_argument(
        "--input",
        "-i",
        default=str(DATA_DIR / DEFAULT_SPLITS_PARQUET_FILE),
        help="Path to concatenated Parquet with 'split' column",
    )
    parser.add_argument(
        "--target",
        "-t",
        default=None,
        help="Target column name (if omitted, inferred among common variants)",
    )
    parser.add_argument(
        "--output-json",
        "-oj",
        default=str(DEFAULT_FEATURE_IMPORTANCE_JSON_PATH),
        help="Destination JSON file for permutation importances",
    )
    parser.add_argument(
        "--output-plot",
        "-op",
        default=str(DEFAULT_FEATURE_IMPORTANCE_PLOT_PATH),
        help="Destination image file for permutation importances plot",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=DEFAULT_PERMUTATION_REPEATS,
        help="Number of permutation repeats",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for model and permutation",
    )
    return parser


def _load_train_val_and_target(input_path: Path, target_option: str | None, random_state: int):
    """Load splits and return (train_df, val_df, target_col)."""
    splits = load_splits_from_parquet(input_path)
    if len(splits) == 2:
        train_df, _test_df = splits
        target_col = target_option or _infer_target_column(list(train_df.columns))
        train_df, val_df = split_train_test(
            train_df, target_column=target_col, random_state=random_state
        )
    else:
        train_df, val_df, _test_df = splits
        target_col = target_option or _infer_target_column(list(train_df.columns))
    return train_df, val_df, target_col


def _compute_and_persist(
    *,
    train_df,
    val_df,
    target_col: str,
    output_json_path: Path,
    output_plot_path: Path,
    n_repeats: int,
    random_state: int,
) -> None:
    """Compute permutation importance, then save JSON and plot."""
    X_train, y_train = split_features_target(train_df, target_col)
    X_val, y_val = split_features_target(val_df, target_col)
    importance_df = compute_random_forest_permutation_importance(
        X_train,
        y_train,
        X_val,
        y_val,
        n_repeats=int(n_repeats),
        random_state=int(random_state),
        n_jobs=DEFAULT_PERMUTATION_N_JOBS,
    )
    save_feature_importances_to_json(importance_df, str(output_json_path))
    try:
        plot_feature_importances(importance_df, str(output_plot_path), top_n=None)
    except RuntimeError as exc:
        if "matplotlib" in str(exc):
            LOGGER.warning("Plot skipped: %s", exc)
        else:
            raise


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    input_path = Path(args.input)
    output_json_path = Path(args.output_json)
    output_plot_path = Path(args.output_plot)
    train_df, val_df, target_col = _load_train_val_and_target(
        input_path, args.target, args.random_state
    )
    _compute_and_persist(
        train_df=train_df,
        val_df=val_df,
        target_col=target_col,
        output_json_path=output_json_path,
        output_plot_path=output_plot_path,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
