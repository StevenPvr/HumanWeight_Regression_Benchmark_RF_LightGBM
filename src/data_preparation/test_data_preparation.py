"""Unit tests for data preparation module."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
# WHY: enable direct pytest invocation without packaging by exposing project root for absolute imports
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


import pytest
import pandas as pd
import numpy as np
from src.data_preparation.data_preparation import (
    encode_categorical_variables,
    shuffle_data,
    split_train_val_test,
)
from src.utils import save_splits_with_marker, save_label_encoders_mappings
from src.utils import proportional_allocation_indices
from src.data_preparation.main import run_preparation_pipeline
from src.constants import TARGET_COLUMN


def test_encode_categorical_variables() -> None:
    """Test encoding of categorical variables with LabelEncoder."""
    # Create mock data with categorical and numerical columns
    mock_data = {
        'age': [25, 30, 35, 40],
        'gender': ['Male', 'Female', 'Male', 'Female'],
        'workout-type': ['Cardio', 'HIIT', 'Strength', 'Cardio'],
        'diet-type': ['Vegan', 'Paleo', 'Vegan', 'Vegetarian'],
        'weight-(kg)': [70.5, 65.2, 80.1, 55.8]
    }
    df_mock = pd.DataFrame(mock_data)

    # Apply encoding
    df_encoded, encoders = encode_categorical_variables(df_mock)

    # Check that categorical columns are encoded
    assert df_encoded['gender'].dtype in [np.int32, np.int64]
    assert df_encoded['workout-type'].dtype in [np.int32, np.int64]
    assert df_encoded['diet-type'].dtype in [np.int32, np.int64]

    # Check that numerical columns remain unchanged
    assert df_encoded['age'].dtype == df_mock['age'].dtype
    assert df_encoded['weight-(kg)'].dtype == df_mock['weight-(kg)'].dtype

    # Check that encoders are returned for categorical columns
    assert 'gender' in encoders
    assert 'workout-type' in encoders
    assert 'diet-type' in encoders
    assert 'age' not in encoders
    assert 'weight-(kg)' not in encoders

    # Check that encoding is reversible
    assert list(encoders['gender'].inverse_transform(df_encoded['gender'])) == list(df_mock['gender'])

    # Check that shape is preserved
    assert df_encoded.shape == df_mock.shape


def test_encode_categorical_variables_no_categorical() -> None:
    """Test encoding with DataFrame containing no categorical columns."""
    # Create mock data with only numerical columns
    mock_data = {
        'age': [25, 30, 35],
        'weight': [70.5, 65.2, 80.1],
        'height': [175, 168, 182]
    }
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
    mock_data = {
        'id': [1, 2, 3, 4, 5],
        'value': [10, 20, 30, 40, 50]
    }
    df_mock = pd.DataFrame(mock_data)

    # Shuffle with same random_state twice
    df_shuffled_1 = shuffle_data(df_mock, random_state=123)
    df_shuffled_2 = shuffle_data(df_mock, random_state=123)

    # Check that both shuffles produce identical results
    pd.testing.assert_frame_equal(df_shuffled_1, df_shuffled_2)


def test_shuffle_data_preserves_row_count() -> None:
    """Test that shuffle preserves the number of rows."""
    # Create mock data
    mock_data = {
        'age': [25, 30, 35, 40, 45],
        'weight': [70.5, 65.2, 80.1, 55.8, 90.3]
    }
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
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    }
    df_mock = pd.DataFrame(mock_data)

    # Shuffle data
    df_shuffled = shuffle_data(df_mock, random_state=123)

    # Check that all values are preserved (sorted comparison)
    assert sorted(df_shuffled['id'].tolist()) == sorted(df_mock['id'].tolist())
    assert sorted(df_shuffled['value'].tolist()) == sorted(df_mock['value'].tolist())


def test_shuffle_data_changes_order() -> None:
    """Test that shuffle actually changes the order of rows."""
    # Create mock data with sequential IDs
    mock_data = {
        'id': list(range(100)),
        'value': list(range(0, 200, 2))
    }
    df_mock = pd.DataFrame(mock_data)

    # Shuffle data
    df_shuffled = shuffle_data(df_mock, random_state=123)

    # Check that order has changed (very unlikely to remain identical with 100 rows)
    order_changed = not df_shuffled['id'].equals(df_mock['id'])
    assert order_changed


def test_shuffle_data_resets_index() -> None:
    """Test that shuffle resets index to sequential integers."""
    # Create mock data
    mock_data = {
        'value': [10, 20, 30, 40, 50]
    }
    df_mock = pd.DataFrame(mock_data)
    # Set non-sequential index
    df_mock.index = pd.Index([5, 3, 1, 4, 2])

    # Shuffle data
    df_shuffled = shuffle_data(df_mock, random_state=123)

    # Check that index is reset to sequential integers starting from 0
    expected_index = list(range(len(df_shuffled)))
    assert df_shuffled.index.tolist() == expected_index


def test_split_train_val_test_basic_sizes_and_reproducibility() -> None:
    """Train/val/test sizes follow 60/20/20 on 100 rows and are reproducible."""
    # Build deterministic DataFrame of 100 rows
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        'weight-(kg)': rng.normal(loc=70.0, scale=10.0, size=100),
        'age': rng.randint(18, 65, size=100),
        'gender': rng.choice(['Male', 'Female'], size=100),
    })

    # Split twice with same seed
    t1, v1, te1 = split_train_val_test(df, target_column='weight-(kg)', random_state=123, n_strata=10)
    t2, v2, te2 = split_train_val_test(df, target_column='weight-(kg)', random_state=123, n_strata=10)

    # Check sizes are exactly 60/20/20 for 100 rows
    assert len(t1) == 60
    assert len(v1) == 20
    assert len(te1) == 20

    # Check reproducibility
    pd.testing.assert_frame_equal(t1, t2)
    pd.testing.assert_frame_equal(v1, v2)
    pd.testing.assert_frame_equal(te1, te2)


def test_split_train_val_test_preserves_total_and_tv_stratification() -> None:
    """Total rows preserved and train/val are stratified without using test target."""
    rng = np.random.RandomState(7)
    weights = np.concatenate([
        rng.normal(60, 5, 40),
        rng.normal(70, 5, 40),
        rng.normal(80, 5, 40),
    ])
    df = pd.DataFrame({
        'weight-(kg)': weights,
        'x': rng.randn(120),
    })

    train_df, val_df, test_df = split_train_val_test(
        df, target_column='weight-(kg)', random_state=123, n_strata=8
    )

    # Total preserved and test size correct
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert len(test_df) == int(round(0.2 * len(df)))

    # Derive bin edges from train+val only (no leakage)
    tv_df = pd.concat([train_df, val_df], ignore_index=True)
    _, bin_edges = pd.qcut(tv_df['weight-(kg)'], q=8, duplicates='drop', retbins=True)
    # Pylance type: convert to a standard Python list[float] (Sequence[float])
    bin_edges_list = [float(x) for x in bin_edges]
    # WHY: Use pd.Categorical(...).codes instead of Series.cat.codes to satisfy static type checkers (Pyright)
    strata_tv = np.asarray(pd.Categorical(pd.cut(tv_df['weight-(kg)'], bins=bin_edges_list, include_lowest=True)).codes)
    strata_train = np.asarray(pd.Categorical(pd.cut(train_df['weight-(kg)'], bins=bin_edges_list, include_lowest=True)).codes)
    strata_val = np.asarray(pd.Categorical(pd.cut(val_df['weight-(kg)'], bins=bin_edges_list, include_lowest=True)).codes)

    # Within each tv bin, allocations should follow 75/25 rounding from remaining data
    unique_bins = np.unique(strata_tv[~pd.isna(strata_tv)])
    for bin_id in unique_bins:
        n_bin = int(np.sum(strata_tv == bin_id))
        ideal = np.array([n_bin * 0.75, n_bin * 0.25])
        floors = np.floor(ideal).astype(int)
        remainder = int(n_bin - floors.sum())
        frac_parts = ideal - floors
        order = np.argsort(-frac_parts)
        alloc = floors.copy()
        for i in range(remainder):
            alloc[order[i]] += 1
        exp_train, exp_val = alloc.tolist()

        ct_t = int(np.sum(strata_train == bin_id))
        ct_v = int(np.sum(strata_val == bin_id))
        assert ct_t == exp_train
        assert ct_v == exp_val


def test_split_train_val_test_errors_and_empty() -> None:
    """Missing target raises error and empty df returns empty splits."""
    # Error when target missing
    with pytest.raises(ValueError):
        split_train_val_test(pd.DataFrame({'a': [1, 2]}), target_column='weight-(kg)')

    # Empty DataFrame → empty splits
    e_train, e_val, e_test = split_train_val_test(pd.DataFrame(), target_column='weight-(kg)')
    assert e_train.empty and e_val.empty and e_test.empty


def test_save_splits_with_marker_csv(tmp_path: Path) -> None:
    """CSV output contains all rows with correct split labels."""
    train = pd.DataFrame({"id": [1, 2], "x": [10, 20]})
    val = pd.DataFrame({"id": [3, 4], "x": [30, 40]})
    test = pd.DataFrame({"id": [5, 6], "x": [50, 60]})

    csv_path = tmp_path / "joined.csv"
    parquet_path = tmp_path / "joined.parquet"

    save_splits_with_marker(train, val, test, csv_path, parquet_path)

    assert csv_path.exists()
    df_csv = pd.read_csv(csv_path)
    assert "split" in df_csv.columns
    assert len(df_csv) == len(train) + len(val) + len(test)
    assert set(df_csv["split"]) == {"train", "val", "test"}


def test_save_splits_with_marker_parquet_optional(tmp_path: Path) -> None:
    """Parquet write/read works when engine is available; otherwise skip."""
    train = pd.DataFrame({"id": [1], "x": [10]})
    val = pd.DataFrame({"id": [2], "x": [20]})
    test = pd.DataFrame({"id": [3], "x": [30]})

    csv_path = tmp_path / "j.csv"
    parquet_path = tmp_path / "j.parquet"

    try:
        save_splits_with_marker(train, val, test, csv_path, parquet_path)
        df_pq = pd.read_parquet(parquet_path)  # may raise if engine missing
    except Exception as exc:  # engine may be missing in some environments
        pytest.skip(f"Parquet engine not available: {exc}")

    assert len(df_pq) == 3
    assert set(df_pq["split"]) == {"train", "val", "test"}


def test_save_label_encoders_mappings(tmp_path: Path) -> None:
    """JSON and CSV mappings reflect LabelEncoder class order and indices."""
    df = pd.DataFrame({
        'gender': ['Male', 'Female', 'Male'],
        'workout-type': ['Cardio', 'Strength', 'Cardio'],
        'x': [1, 2, 3],
    })

    df_enc, encoders = encode_categorical_variables(df)
    json_path = tmp_path / 'encoders.json'
    csv_path = tmp_path / 'encoders.csv'

    save_label_encoders_mappings(encoders, json_path, csv_path)

    # JSON assertions
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    assert 'gender' in data and 'workout-type' in data
    for col, _le in encoders.items():
        # LabelEncoder sorts classes lexicographically by default
        classes = sorted([str(x) for x in df[col].unique()])
        expected_map = {cls: i for i, cls in enumerate(classes)}
        assert data[col]['classes'] == classes
        assert data[col]['mapping'] == expected_map

    # CSV assertions
    df_map = pd.read_csv(csv_path)
    assert set(df_map.columns) == {'column', 'original_value', 'encoded_value'}
    total_rows_expected = sum(len(v['classes']) for v in data.values())
    assert len(df_map) == total_rows_expected
    # Check a couple of rows exist
    assert any((df_map['column'] == 'gender') & (df_map['original_value'] == 'Female'))
    assert any((df_map['column'] == 'gender') & (df_map['original_value'] == 'Male'))


def test_save_label_encoders_mappings_empty(tmp_path: Path) -> None:
    """Empty encoders produce empty JSON object and empty CSV (header only)."""
    json_path = tmp_path / 'empty.json'
    csv_path = tmp_path / 'empty.csv'
    save_label_encoders_mappings({}, str(json_path), str(csv_path))

    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert data == {}

    df_csv = pd.read_csv(csv_path)
    assert list(df_csv.columns) == ['column', 'original_value', 'encoded_value']
    assert len(df_csv) == 0


def test_proportional_allocation_indices() -> None:
    """Per-group allocation honors proportions with deterministic rounding."""
    rng = np.random.RandomState(123)
    # Two groups with different sizes to test rounding behavior
    groups = {
        0: np.arange(0, 8),   # size 8 → 6/2 with 0.75/0.25
        1: np.arange(10, 21), # size 11 → 8/3 with 0.75/0.25 (remainder to larger frac)
    }
    first, second = proportional_allocation_indices(groups, (0.75, 0.25), rng=rng)

    # Check partition properties
    assert len(set(first).intersection(second)) == 0
    all_idx = sorted(first + second)
    assert all_idx == sorted(list(groups[0]) + list(groups[1]))

    # Verify counts per group
    first_0 = [i for i in first if 0 <= i < 8]
    second_0 = [i for i in second if 0 <= i < 8]
    assert len(first_0) == 6 and len(second_0) == 2

    first_1 = [i for i in first if 10 <= i < 21]
    second_1 = [i for i in second if 10 <= i < 21]
    assert len(first_1) == 8 and len(second_1) == 3


def test_run_preparation_pipeline_smoke(tmp_path: Path) -> None:
    """End-to-end pipeline creates outputs and respects split contract."""
    # Build a tiny synthetic dataset with categorical + numeric target
    df = pd.DataFrame({
        TARGET_COLUMN: [70.0, 65.0, 80.0, 55.0, 75.0, 68.0, 72.0, 90.0],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'diet-type': ['Vegan', 'Paleo', 'Vegan', 'Vegetarian', 'Paleo', 'Vegan', 'Vegetarian', 'Paleo'],
    })

    input_csv = tmp_path / 'input.csv'
    output_csv = tmp_path / 'splits.csv'
    output_parquet = tmp_path / 'splits.parquet'
    enc_json = tmp_path / 'enc.json'
    enc_csv = tmp_path / 'enc.csv'
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
            n_strata=4,
        )
    except Exception as exc:
        # Parquet may be unavailable; if the error hints at parquet engine, skip
        if 'Parquet' in str(exc) or 'pyarrow' in str(exc) or 'fastparquet' in str(exc):
            pytest.skip(f"Parquet engine not available: {str(exc)}")
        raise

    assert output_csv.exists()
    assert enc_json.exists() and enc_csv.exists()

    df_out = pd.read_csv(output_csv)
    assert 'split' in df_out.columns
    assert set(df_out['split']) == {'train', 'val', 'test'}
    assert len(df_out) == len(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
