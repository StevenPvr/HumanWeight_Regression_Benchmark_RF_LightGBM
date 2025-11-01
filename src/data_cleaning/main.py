"""Main pipeline to execute data cleaning operations."""

from __future__ import annotations

from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from matplotlib import pyplot as plt

from src.data_cleaning.data_cleaning import (
    binarize_physical_exercise,
    normalize_weigh_lifestyle_columns,
    save_cleaned_dataset,
)
from src.utils import get_logger, round_columns


LOGGER = get_logger(__name__)


def main() -> None:
    """Execute the complete data cleaning pipeline from raw to final dataset."""

    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / "data" / "dataset_cleaned.csv"

    # Loading the dataset that was already cleaned during EDA avoids redundant preprocessing.
    LOGGER.info("Loading dataset from %s", input_path)
    df = pd.read_csv(input_path)
    LOGGER.info("Loaded %d rows and %d columns", len(df), len(df.columns))

    df = normalize_weigh_lifestyle_columns(df)
    LOGGER.info("Column names normalized")

    df = binarize_physical_exercise(df)
    LOGGER.info("Physical exercise column binarized")

    # Round selected engineered metrics to 1 decimal for readability
    df = round_columns(
        df,
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
    LOGGER.info("Selected numeric columns rounded to 1 decimal place")

    # Visualizing class balance ensures the binarization remains aligned with modeling needs.

    counts = df["physical-exercise"].value_counts().sort_index()
    plots_dir = project_root / "plots" / "distribution"
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
    LOGGER.info("Distribution plot saved to %s", output_plot)

    save_cleaned_dataset(df)
    LOGGER.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
