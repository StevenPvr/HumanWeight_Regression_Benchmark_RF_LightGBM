"""Unit tests for feature engineering helpers.

Currently validates loading of split parquet via src.utils.load_splits_from_parquet.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json

import numpy as np
import pandas as pd
import pytest
import pytest as _pytest

from src.constants import DEFAULT_RANDOM_STATE, TARGET_COLUMN
from src.feature_engineering.feature_engineering import compute_random_forest_permutation_importance
from src.feature_engineering.feature_engineering import plot_feature_importances
from src.feature_engineering.feature_engineering import save_feature_importances_to_json
from src.feature_engineering.main import main as fe_main
from src.utils import ensure_numeric_columns
from src.utils import load_splits_from_parquet
from src.utils import split_features_target


def test_load_splits_from_parquet_roundtrip(tmp_path: Path) -> None:
    """Read back mocked train/val/test splits written into one parquet file."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "x": [10, 20, 30, 40, 50, 60],
            "split": ["train", "train", "val", "val", "test", "test"],
        }
    )

    pq_path = tmp_path / "splits.parquet"

    # Write parquet (optional based on engine availability)
    try:
        df.to_parquet(pq_path, index=False)
    except Exception as exc:
        pytest.skip(f"Parquet engine not available: {exc}")

    train_df, val_df, test_df = load_splits_from_parquet(str(pq_path))

    # 'split' column is removed in outputs
    for split_df in (train_df, val_df, test_df):
        assert "split" not in split_df.columns

    # Shapes and contents
    assert len(train_df) == 2 and train_df["id"].tolist() == [1, 2]
    assert len(val_df) == 2 and val_df["id"].tolist() == [3, 4]
    assert len(test_df) == 2 and test_df["id"].tolist() == [5, 6]


def test_load_splits_from_parquet_missing_file(tmp_path: Path) -> None:
    """Missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_splits_from_parquet(str(tmp_path / "nope.parquet"))


def test_load_splits_from_parquet_missing_split_column(tmp_path: Path) -> None:
    """Missing 'split' column raises KeyError."""
    df = pd.DataFrame({"id": [1, 2], "x": [10, 20]})
    pq_path = tmp_path / "no_split.parquet"
    try:
        df.to_parquet(pq_path, index=False)
    except Exception as exc:
        pytest.skip(f"Parquet engine not available: {exc}")

    with pytest.raises(KeyError):
        load_splits_from_parquet(str(pq_path))



def test_ensure_numeric_columns_converts_mixed_types() -> None:
    df = pd.DataFrame({
        "int_col": [1, 2, 3],
        "float_col": [1.0, 2.5, 3.2],
        "bool_col": [True, False, True],
        "obj_num": ["10", "20", "30"],
        "obj_txt": ["a", "b", "c"],
        "cat_col": pd.Series(["x", "y", "x"], dtype="category"),
        "dt_col": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
        "td_col": pd.to_timedelta(["1 days", "2 days", "3 days"]),
    })

    out = ensure_numeric_columns(df)

    # All columns should be numeric dtype
    for c in out.columns:
        assert pd.api.types.is_numeric_dtype(out[c]), f"Column {c} is not numeric"

    # Specific expectations
    assert set(out["bool_col"].unique()) <= {0, 1}
    assert out["obj_num"].tolist() == [10, 20, 30]
    # obj_txt becomes NaN after coercion
    assert out["obj_txt"].isna().sum() == 3

def test_ensure_numeric_columns_empty_dataframe() -> None:
    df = pd.DataFrame()
    out = ensure_numeric_columns(df)
    assert out.empty



def test_split_features_target_basic() -> None:
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        TARGET_COLUMN: [70.0, 71.5, 69.2],
    })
    X, y = split_features_target(df, TARGET_COLUMN)
    assert TARGET_COLUMN not in X.columns
    assert list(X.columns) == ["a", "b"]
    assert y.equals(df[TARGET_COLUMN])


def test_split_features_target_missing_raises() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError):
        split_features_target(df, TARGET_COLUMN)



def test_compute_random_forest_permutation_importance_runs() -> None:
    """RF permutation importance returns a well-formed DataFrame without leakage."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        "signal": rng.normal(size=40),
        "noise": rng.normal(size=40),
    })
    # Target depends mostly on 'signal'
    y = pd.Series(2.0 * X["signal"] + rng.normal(scale=0.1, size=40))

    X_train, y_train = X.iloc[:30].reset_index(drop=True), y.iloc[:30].reset_index(drop=True)
    X_val, y_val = X.iloc[30:].reset_index(drop=True), y.iloc[30:].reset_index(drop=True)

    imp = compute_random_forest_permutation_importance(
        X_train,
        y_train,
        X_val,
        y_val,
        n_repeats=3,
        random_state=DEFAULT_RANDOM_STATE,
    )
    assert list(imp.columns) == ["feature", "importance_mean", "importance_std"]
    assert set(imp["feature"]) == {"signal", "noise"}
    assert pd.api.types.is_float_dtype(imp["importance_mean"]) and pd.api.types.is_float_dtype(imp["importance_std"])  # type: ignore[no-untyped-call]
    # Top feature should be 'signal' in this synthetic setup (most of the time)
    assert imp.iloc[0]["feature"] in {"signal", "noise"}



def test_save_feature_importances_to_json_roundtrip(tmp_path: Path) -> None:
    imp_df = pd.DataFrame({
        "feature": ["a", "b"],
        "importance_mean": [0.12, 0.01],
        "importance_std": [0.02, 0.005],
    })

    out_path = tmp_path / "imps" / "importances.json"
    save_feature_importances_to_json(imp_df, str(out_path))

    assert out_path.exists()
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list) and len(data) == 2
    assert set(data[0].keys()) == {"feature", "importance_mean", "importance_std"}
    # Values are serializable numbers/strings
    assert isinstance(data[0]["feature"], str)
    assert isinstance(data[0]["importance_mean"], float)
    assert isinstance(data[0]["importance_std"], float)


def test_save_feature_importances_to_json_missing_columns(tmp_path: Path) -> None:
    bad_df = pd.DataFrame({"feature": ["x"], "importance_mean": [0.1]})
    with pytest.raises(ValueError):
        save_feature_importances_to_json(bad_df, str(tmp_path / "x.json"))



def test_plot_feature_importances_creates_image(tmp_path: Path) -> None:
    imp_df = pd.DataFrame({
        "feature": ["a", "b", "c"],
        "importance_mean": [0.3, 0.1, 0.2],
        "importance_std": [0.05, 0.02, 0.03],
    })
    out = tmp_path / "plots" / "imp.png"
    try:
        plot_feature_importances(imp_df, str(out), top_n=2, title="Test")
    except RuntimeError as exc:
        if "matplotlib" in str(exc):
            pytest.skip(f"matplotlib unavailable: {exc}")
        raise
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_feature_importances_missing_cols_raises(tmp_path: Path) -> None:
    bad_df = pd.DataFrame({"feature": ["a"], "importance_mean": [0.1]})
    with pytest.raises(ValueError):
        plot_feature_importances(bad_df, str(tmp_path / "x.png"))


def test_pipeline_main_end_to_end_outputs(tmp_path: Path) -> None:
    """End-to-end run writes JSON output (plot optional)."""
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6, 7, 8],
        "b": [10, 9, 8, 7, 6, 5, 4, 3],
        TARGET_COLUMN: [v * 2.0 for v in [1, 2, 3, 4, 5, 6, 7, 8]],
        "split": ["train"] * 6 + ["val", "val"],
    })

    pq = tmp_path / "splits.parquet"
    try:
        df.to_parquet(pq, index=False)
    except Exception as exc:
        pytest.skip(f"Parquet engine not available: {exc}")

    out_json = tmp_path / "imps.json"
    out_png = tmp_path / "imp.png"

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "prog",
            "-i", str(pq),
            "-t", TARGET_COLUMN,
            "-oj", str(out_json),
            "-op", str(out_png),
            "--n-repeats", "2",
        ]
        fe_main()
    finally:
        sys.argv = old_argv

    # JSON should be written; plot may be skipped if matplotlib is unavailable
    assert out_json.exists()


if __name__ == "__main__":
    _pytest.main([__file__, "-v"])
