"""Data cleaning utilities for the Weigh Lifestyle project.

The helpers here favor deterministic, copy-on-write transformations to keep
pipelines composable and debuggable. Where historical or multilingual column
labels occur, we normalize to a small set of canonical slugs to eliminate
downstream ambiguity.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from src.utils import get_logger


LOGGER = get_logger(__name__)


def normalize_weigh_lifestyle_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to canonical, lowercase, hyphenated slugs.

    Rationale: standardizing to stable slugs avoids brittle downstream code that
    depends on capitalization, spacing, or mixed-language spellings (e.g., the
    French "exercice"). We keep normalization minimal (KISS) and only enforce
    canonical remaps when there is known ambiguity.

    Parameters
    ----------
    dataset: pd.DataFrame
        Input data whose columns need normalization.

    Returns
    -------
    pd.DataFrame
        Copy of ``dataset`` with sanitized column names.
    """

    # Copying avoids mutating the caller's frame so composed pipelines stay deterministic.
    normalized_df = dataset.copy()
    normalized_df.columns = (
        normalized_df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s_]+", "-", regex=True)
        .str.replace(r"-+", "-", regex=True)
        .str.strip("-")
    )

    # WHY: Map known multilingual/typo variants to a canonical slug to keep
    # references consistent across notebooks and code (DRY across the project).
    canonical_map = {
        "physical-exercice": "physical-exercise",  # French/typo -> English canonical
    }
    normalized_df.rename(columns=canonical_map, inplace=True)

    return normalized_df



def binarize_physical_exercise(dataset: pd.DataFrame) -> pd.DataFrame:
    """Convert the physical exercise column into a binary indicator.

    Parameters
    ----------
    dataset: pd.DataFrame
        Input data containing a normalized physical exercise column.

    Returns
    -------
    pd.DataFrame
        Copy of ``dataset`` with the physical exercise column mapped to ``{0, 1}``.
    """

    # Working on a copy prevents accidental mutation of user-provided frames in pipelines.
    binarized_df = dataset.copy()

    # Enforcing the normalized slug prevents silently accepting mismatched column names.
    possible_columns = (
        "physical-exercise",
    )
    target_column = next((col for col in possible_columns if col in binarized_df.columns), None)
    if target_column is None:
        msg = (
            "DataFrame must include a 'physical-exercise' column after "
            "normalization. Ensure you called normalize_weigh_lifestyle_columns first."
        )
        raise KeyError(msg)

    numeric_series = cast(
        pd.Series,
        pd.to_numeric(binarized_df[target_column], errors="coerce"),
    )
    binarized_df[target_column] = (numeric_series.fillna(0) > 0).astype(int)
    return binarized_df


def save_cleaned_dataset(
    dataset: pd.DataFrame,
    output_name: str = "dataset_cleaned_final",
    output_dir: Path | str | None = None,
) -> None:
    """Save the cleaned dataset in both CSV and Parquet formats.

    Parameters
    ----------
    dataset: pd.DataFrame
        Cleaned data ready for modeling.
    output_name: str
        Base filename without extension (default: "dataset_cleaned_final").
    output_dir: Path | str | None
        Destination directory for the exported files. When ``None`` the files are
        saved in the project's ``data/`` directory.

    Returns
    -------
    None
        Files are written to ``output_dir`` or the default ``data/`` directory.
    """

    # Resolving paths relative to this module keeps the function portable across environments.
    resolved_output_dir = (
        Path(__file__).parent.parent.parent / "data"
        if output_dir is None
        else Path(output_dir)
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = resolved_output_dir / f"{output_name}.csv"
    parquet_path = resolved_output_dir / f"{output_name}.parquet"

    dataset.to_csv(csv_path, index=False)
    LOGGER.info("Saved CSV to %s", csv_path)

    dataset.to_parquet(parquet_path, index=False)
    LOGGER.info("Saved Parquet to %s", parquet_path)
