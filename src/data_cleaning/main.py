"""Main pipeline to execute data cleaning operations."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from src.constants import DATA_DIR, PLOTS_DIR  # noqa: E402
from src.data_cleaning.data_cleaning import (  # noqa: E402
    binarize_physical_exercise,
    normalize_weigh_lifestyle_columns,
    save_cleaned_dataset,
)
from src.utils import get_logger, round_columns  # noqa: E402

LOGGER = get_logger(__name__)


def _source_csv_path() -> Path:
    """Return the input CSV path for the cleaning stage.

    WHY: Keep path construction centralized and import constants for base dirs.
    """
    return DATA_DIR / "dataset_cleaned.csv"


def _apply_cleaning_steps(df: pd.DataFrame) -> pd.DataFrame:
    """Apply normalization, binarization, and rounding to the dataset."""
    out = normalize_weigh_lifestyle_columns(df)
    out = binarize_physical_exercise(out)
    out = round_columns(
        out,
        columns=[
            "fat-percentage",
            "bmi-calc",
            "protein-per-kg",
            "cal-balance",
            "lean-mass-kg",
            "burns-calories-(per-30-min)-bc",
        ],
        decimals=1,
    )
    return out


def _save_distribution_plot(df: pd.DataFrame) -> Path:
    """Create and save the physical-exercise distribution plot.

    Returns the path to the saved PNG file.
    """
    counts = df["physical-exercise"].value_counts().sort_index()
    plots_dir = PLOTS_DIR / "distribution"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_plot = plots_dir / "new_physical_exercise_distribution.png"

    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"])
    ax.set_title("New Physical Exercise Distribution")
    ax.set_xlabel("physical-exercise (0/1)")
    ax.set_ylabel("count")
    ax.set_xticklabels(["0", "1"], rotation=0)
    fig.tight_layout()
    fig.savefig(output_plot, dpi=150)
    plt.close(fig)
    return output_plot


def main() -> None:
    """Execute the complete data cleaning pipeline from raw to final dataset."""
    input_path = _source_csv_path()
    LOGGER.info("Loading dataset from %s", input_path)
    df = pd.read_csv(input_path)
    LOGGER.info("Loaded %d rows and %d columns", len(df), len(df.columns))

    df = _apply_cleaning_steps(df)
    LOGGER.info("Cleaning steps applied (normalize, binarize, round)")

    output_plot = _save_distribution_plot(df)
    LOGGER.info("Distribution plot saved to %s", output_plot)

    save_cleaned_dataset(df)
    LOGGER.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
