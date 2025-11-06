"""Unit tests for data preparation module."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.constants import TARGET_COLUMN
from src.data_preparation.data_preparation import (
    encode_categorical_variables,
    shuffle_data,
    split_train_test,
)
from src.data_preparation.main import run_preparation_pipeline
from src.utils import save_label_encoders_mappings, save_splits_with_marker


@pytest.fixture
def categorical_dataframe() -> pd.DataFrame:
    """Provide a tiny mixed-type dataset for encoding related tests."""

    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40],
            "gender": ["Male", "Female", "Male", "Female"],
            "workout-type": ["Cardio", "HIIT", "Strength", "Cardio"],
            "diet-type": ["Vegan", "Paleo", "Vegan", "Vegetarian"],
            "weight-(kg)": [70.5, 65.2, 80.1, 55.8],
        }
    )


def test_encode_categorical_variables_encodes_object_columns(
    categorical_dataframe: pd.DataFrame,
) -> None:
    """Categorical columns are replaced by OHE columns named with '<col>_<cat>'."""

    encoded, encoders = encode_categorical_variables(categorical_dataframe)

    for column in ["gender", "workout-type", "diet-type"]:
        # Encoder must exist for each categorical column
        assert column in encoders
        # Original column removed and OHE columns present
        assert column not in encoded.columns
        ohe_cols = [c for c in encoded.columns if c.startswith(column + "_")]
        assert ohe_cols, f"Expected one-hot columns for {column}"
        # Values are numeric 0/1
        assert all(pd.api.types.is_numeric_dtype(encoded[c]) for c in ohe_cols)
        assert set(np.unique(encoded[ohe_cols].to_numpy().ravel())).issubset({0.0, 1.0})


def test_encode_categorical_variables_preserves_numeric_columns(
    categorical_dataframe: pd.DataFrame,
) -> None:
    """Numeric columns should retain their dtype and values after encoding."""

    encoded, _ = encode_categorical_variables(categorical_dataframe)

    pd.testing.assert_series_equal(encoded["age"], categorical_dataframe["age"])
    pd.testing.assert_series_equal(encoded["weight-(kg)"], categorical_dataframe["weight-(kg)"])


def test_encode_categorical_variables_inverse_transform_roundtrip(
    categorical_dataframe: pd.DataFrame,
) -> None:
    """OHE blocks invert back to original categories for a column."""

    encoded, encoders = encode_categorical_variables(categorical_dataframe)

    # Build the one-hot matrix for 'gender' and inverse transform
    gender_cols = [c for c in encoded.columns if c.startswith("gender_")]
    assert gender_cols, "Expected OHE columns for 'gender'"
    inv = encoders["gender"].inverse_transform(encoded[gender_cols].to_numpy())
    recovered = np.array(inv).ravel().tolist()
    assert recovered == categorical_dataframe["gender"].tolist()


def test_encode_categorical_variables_no_categorical() -> None:
    """Test encoding with DataFrame containing no categorical columns."""
    # Create mock data with only numerical columns
    mock_data = {"age": [25, 30, 35], "weight": [70.5, 65.2, 80.1], "height": [175, 168, 182]}
    df_mock = pd.DataFrame(mock_data)

    # Apply encoding
    df_encoded, encoders = encode_categorical_variables(df_mock)

    # Check that no encoders are created
    assert len(encoders) == 0

    # Check that DataFrame is unchanged
    pd.testing.assert_frame_equal(df_encoded, df_mock)


def test_encode_categorical_variables_empty_dataframe() -> None:
    """Test encoding with empty DataFrame."""
    df_mock = pd.DataFrame()

    # Apply encoding
    df_encoded, encoders = encode_categorical_variables(df_mock)

    # Check that no encoders are created
    assert len(encoders) == 0

    # Check that DataFrame is empty
    assert df_encoded.empty


def test_shuffle_data_reproducibility() -> None:
    """Test that shuffle is reproducible with same random_state."""
    # Create mock data
    mock_data = {"id": [1, 2, 3, 4, 5], "value": [10, 20, 30, 40, 50]}
    df_mock = pd.DataFrame(mock_data)

    # Shuffle with same random_state twice
    df_shuffled_1 = shuffle_data(df_mock, random_state=123)
    df_shuffled_2 = shuffle_data(df_mock, random_state=123)

    # Check that both shuffles produce identical results
    pd.testing.assert_frame_equal(df_shuffled_1, df_shuffled_2)


def test_shuffle_data_preserves_row_count() -> None:
    """Test that shuffle preserves the number of rows."""
    # Create mock data
    mock_data = {"age": [25, 30, 35, 40, 45], "weight": [70.5, 65.2, 80.1, 55.8, 90.3]}
    df_mock = pd.DataFrame(mock_data)

    # Shuffle data
    df_shuffled = shuffle_data(df_mock, random_state=123)

    # Check that row count is preserved
    assert len(df_shuffled) == len(df_mock)
    assert df_shuffled.shape[0] == df_mock.shape[0]


def test_shuffle_data_preserves_all_values() -> None:
    """Test that shuffle preserves all values without loss."""
    # Create mock data
    mock_data = {
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "value": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    }
    df_mock = pd.DataFrame(mock_data)

    # Shuffle data
    df_shuffled = shuffle_data(df_mock, random_state=123)

    # Check that all values are preserved (sorted comparison)
    assert sorted(df_shuffled["id"].tolist()) == sorted(df_mock["id"].tolist())
    assert sorted(df_shuffled["value"].tolist()) == sorted(df_mock["value"].tolist())


def test_shuffle_data_changes_order() -> None:
    """Test that shuffle actually changes the order of rows."""
    # Create mock data with sequential IDs
    mock_data = {"id": list(range(100)), "value": list(range(0, 200, 2))}
    df_mock = pd.DataFrame(mock_data)

    # Shuffle data
    df_shuffled = shuffle_data(df_mock, random_state=123)

    # Check that order has changed (very unlikely to remain identical with 100 rows)
    order_changed = not df_shuffled["id"].equals(df_mock["id"])
    assert order_changed


def test_shuffle_data_resets_index() -> None:
    """Test that shuffle resets index to sequential integers."""
    # Create mock data
    mock_data = {"value": [10, 20, 30, 40, 50]}
    df_mock = pd.DataFrame(mock_data)
    # Set non-sequential index
    df_mock.index = pd.Index([5, 3, 1, 4, 2])

    # Shuffle data
    df_shuffled = shuffle_data(df_mock, random_state=123)

    # Check that index is reset to sequential integers starting from 0
    expected_index = list(range(len(df_shuffled)))
    assert df_shuffled.index.tolist() == expected_index


def test_split_train_test_basic_sizes_and_reproducibility(
    random_state_factory: Callable[[int | None], np.random.RandomState]
) -> None:
    """Train/test sizes follow 80/20 on 100 rows and are reproducible."""
    # Build deterministic DataFrame of 100 rows
    rng = random_state_factory(42)
    df = pd.DataFrame(
        {
            "weight-(kg)": rng.normal(loc=70.0, scale=10.0, size=100),
            "age": rng.randint(18, 65, size=100),
            "gender": rng.choice(["Male", "Female"], size=100),
        }
    )

    # Split twice with same seed
    t1, te1 = split_train_test(df, target_column="weight-(kg)", random_state=123)
    t2, te2 = split_train_test(df, target_column="weight-(kg)", random_state=123)

    # Check sizes are exactly 80/20 for 100 rows
    assert len(t1) == 80
    assert len(te1) == 20

    # Check reproducibility
    pd.testing.assert_frame_equal(t1, t2)
    pd.testing.assert_frame_equal(te1, te2)


def test_split_train_test_preserves_total(
    random_state_factory: Callable[[int | None], np.random.RandomState]
) -> None:
    rng = random_state_factory(7)
    df = pd.DataFrame(
        {
            "weight-(kg)": rng.normal(70, 10, 120),
            "x": rng.randn(120),
        }
    )
    train_df, test_df = split_train_test(df, target_column="weight-(kg)", random_state=123)
    assert len(train_df) + len(test_df) == len(df)
    assert len(test_df) == int(round(0.2 * len(df)))


def test_split_train_test_errors_and_empty() -> None:
    """Missing target raises error and empty df returns empty splits."""
    # Error when target missing
    with pytest.raises(ValueError):
        split_train_test(pd.DataFrame({"a": [1, 2]}), target_column="weight-(kg)")

    # Empty DataFrame → empty splits
    e_train, e_test = split_train_test(pd.DataFrame(), target_column="weight-(kg)")
    assert e_train.empty and e_test.empty


def test_save_splits_with_marker_csv(tmp_path: Path) -> None:
    """CSV output contains all rows with correct split labels."""
    train = pd.DataFrame({"id": [1, 2], "x": [10, 20]})
    test = pd.DataFrame({"id": [3, 4], "x": [30, 40]})

    csv_path = tmp_path / "joined.csv"
    parquet_path = tmp_path / "joined.parquet"

    save_splits_with_marker(train, test, csv_path, parquet_path)

    assert csv_path.exists()
    df_csv = pd.read_csv(csv_path)
    assert "split" in df_csv.columns
    assert len(df_csv) == len(train) + len(test)
    assert set(df_csv["split"]) == {"train", "test"}


def test_save_splits_with_marker_parquet_optional(tmp_path: Path) -> None:
    """Parquet write/read works when engine is available; otherwise skip."""
    train = pd.DataFrame({"id": [1], "x": [10]})
    test = pd.DataFrame({"id": [2], "x": [20]})

    csv_path = tmp_path / "j.csv"
    parquet_path = tmp_path / "j.parquet"

    try:
        save_splits_with_marker(train, test, csv_path, parquet_path)
        df_pq = pd.read_parquet(parquet_path)  # may raise if engine missing
    except Exception as exc:  # engine may be missing in some environments
        pytest.skip(f"Parquet engine not available: {exc}")

    assert len(df_pq) == 2
    assert set(df_pq["split"]) == {"train", "test"}


def test_save_label_encoders_mappings_json(tmp_path: Path) -> None:
    """JSON mappings reflect OneHotEncoder categories and feature names."""
    df = pd.DataFrame(
        {
            "gender": ["Male", "Female", "Male"],
            "workout-type": ["Cardio", "Strength", "Cardio"],
            "x": [1, 2, 3],
        }
    )

    df_enc, encoders = encode_categorical_variables(df)
    json_path = tmp_path / "encoders.json"
    csv_path = tmp_path / "encoders.csv"  # not used here but required by signature

    save_label_encoders_mappings(encoders, json_path, csv_path)

    # JSON assertions
    import json

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "gender" in data and "workout-type" in data
    for col, _ohe in encoders.items():
        # OneHotEncoder orders categories lexicographically for strings
        classes = sorted([str(x) for x in df[col].unique()])
        expected_feature_names = [f"{col}_{c}" for c in classes]
        assert data[col]["categories"] == classes
        assert data[col]["feature_names"] == expected_feature_names
        assert data[col]["type"] == "OneHotEncoder"
        # Mapping from category -> feature name
        assert data[col]["mapping"] == {c: f for c, f in zip(classes, expected_feature_names)}


def test_save_label_encoders_mappings_csv(tmp_path: Path) -> None:
    """CSV mapping contains column/category to one-hot feature mapping."""
    df = pd.DataFrame(
        {
            "gender": ["Male", "Female", "Male"],
            "workout-type": ["Cardio", "Strength", "Cardio"],
            "x": [1, 2, 3],
        }
    )

    _, encoders = encode_categorical_variables(df)
    json_path = tmp_path / "encoders.json"
    csv_path = tmp_path / "encoders.csv"

    save_label_encoders_mappings(encoders, json_path, csv_path)

    import json

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df_map = pd.read_csv(csv_path)
    assert set(df_map.columns) == {"column", "original_value", "encoded_value"}
    total_rows_expected = sum(len(v["categories"]) for v in data.values())
    assert len(df_map) == total_rows_expected
    # Check a couple of rows exist
    assert any((df_map["column"] == "gender") & (df_map["encoded_value"] == "gender_Female"))
    assert any((df_map["column"] == "gender") & (df_map["encoded_value"] == "gender_Male"))


def test_save_label_encoders_mappings_empty(tmp_path: Path) -> None:
    """Empty encoders produce empty JSON object and empty CSV (header only)."""
    json_path = tmp_path / "empty.json"
    csv_path = tmp_path / "empty.csv"
    save_label_encoders_mappings({}, json_path, csv_path)

    import json

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data == {}

    df_csv = pd.read_csv(csv_path)
    assert list(df_csv.columns) == ["column", "original_value", "encoded_value"]
    assert len(df_csv) == 0


def test_run_preparation_pipeline_smoke(tmp_path: Path) -> None:
    """End-to-end pipeline creates outputs and respects split contract."""
    # Build a tiny synthetic dataset with categorical + numeric target
    df = pd.DataFrame(
        {
            TARGET_COLUMN: [70.0, 65.0, 80.0, 55.0, 75.0, 68.0, 72.0, 90.0],
            "gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
            "diet-type": [
                "Vegan",
                "Paleo",
                "Vegan",
                "Vegetarian",
                "Paleo",
                "Vegan",
                "Vegetarian",
                "Paleo",
            ],
        }
    )

    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "splits.csv"
    output_parquet = tmp_path / "splits.parquet"
    enc_json = tmp_path / "enc.json"
    enc_csv = tmp_path / "enc.csv"
    df.to_csv(input_csv, index=False)

    try:
        run_preparation_pipeline(
            input_path=input_csv,
            output_csv=output_csv,
            output_parquet=output_parquet,
            encoders_json=enc_json,
            encoders_csv=enc_csv,
            target_column=TARGET_COLUMN,
            random_state=123,
        )
    except Exception as exc:
        # Parquet may be unavailable; if the error hints at parquet engine, skip
        if "Parquet" in str(exc) or "pyarrow" in str(exc) or "fastparquet" in str(exc):
            pytest.skip(f"Parquet engine not available: {str(exc)}")
        raise

    assert output_csv.exists()
    assert enc_json.exists() and enc_csv.exists()

    df_out = pd.read_csv(output_csv)
    assert "split" in df_out.columns
    assert set(df_out["split"]) == {"train", "test"}
    assert len(df_out) == len(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
