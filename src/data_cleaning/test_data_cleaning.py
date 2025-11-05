"""Tests for data cleaning helpers."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pathlib import Path

import pandas as pd
import pytest

from src.data_cleaning.data_cleaning import (
    binarize_physical_exercise,
    normalize_weigh_lifestyle_columns,
    save_cleaned_dataset,
)


def test_normalize_weigh_lifestyle_columns() -> None:
    """It should lowercase names, add hyphens, and keep the source untouched."""

    # Using a tiny synthetic frame isolates the column formatting behavior under test.
    raw_df = pd.DataFrame(
        data=[
            [0, 3, 2.5],
            [1, 2, 2.0],
        ],
        columns=pd.Index([" Physical exercice ", "Daily meals frequency", "Water   Intake_Liters"]),
    )

    normalized_df = normalize_weigh_lifestyle_columns(raw_df)

    # Expect hyphenated lowercase names to guarantee consistent downstream references.
    assert normalized_df.columns.tolist() == [
        "physical-exercise",  # canonical English slug (from French 'exercice')
        "daily-meals-frequency",
        "water-intake-liters",
    ]
    # The original DataFrame must remain unchanged to avoid surprising side effects for callers.
    assert raw_df.columns.tolist() == [
        " Physical exercice ",
        "Daily meals frequency",
        "Water   Intake_Liters",
    ]


def test_binarize_physical_exercise() -> None:
    """It should keep zeros and map positive values to one without mutating the input."""

    # Keeping a copy of the original values guards against in-place side effects during binarization.
    raw_df = pd.DataFrame({"physical-exercise": [0, 0.1, 2, -5]})
    original_values = raw_df["physical-exercise"].tolist()

    binarized_df = binarize_physical_exercise(raw_df)

    assert binarized_df["physical-exercise"].tolist() == [0, 1, 1, 0]
    assert raw_df["physical-exercise"].tolist() == original_values


def test_binarize_with_english_exercise_name() -> None:
    """It should binarize when the column normalizes to 'physical-exercise'."""

    df = pd.DataFrame({"Physical exercise": [0, 0.2, 2, -1]})
    df = normalize_weigh_lifestyle_columns(df)

    # Checking the slug guards against downstream KeyErrors if normalization ever changes.
    assert "physical-exercise" in df.columns

    out = binarize_physical_exercise(df)
    assert out["physical-exercise"].tolist() == [0, 1, 1, 0]


def test_save_cleaned_dataset(tmp_path: Path, tiny_weight_dataframe: pd.DataFrame) -> None:
    """It should write both CSV and Parquet files with matching content."""

    # Using a temporary directory isolates file IO from the actual data folder during testing.
    output_dir = tmp_path / "exports"
    test_output_name = "test_output"

    save_cleaned_dataset(
        tiny_weight_dataframe,
        output_name=test_output_name,
        output_dir=output_dir,
    )

    csv_path = output_dir / f"{test_output_name}.csv"
    parquet_path = output_dir / f"{test_output_name}.parquet"

    # Verifying both formats persist data correctly guards against silent format corruption.
    loaded_csv = pd.read_csv(csv_path)
    loaded_parquet = pd.read_parquet(parquet_path)

    pd.testing.assert_frame_equal(loaded_csv, tiny_weight_dataframe)
    pd.testing.assert_frame_equal(loaded_parquet, tiny_weight_dataframe)


if __name__ == "__main__":

    # Using pytest.main() provides proper test reporting, verbosity, and exit codes.
    pytest.main([__file__, "-v"])
