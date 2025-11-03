from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    from src.utils import (
        read_best_params,
        save_training_results,
        fit_label_encoders_on_train,
        assert_no_overlap_between_train_and_test,
        round_columns,
        prepare_train_val_numeric_splits,
    )
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.utils import (
        read_best_params,
        save_training_results,
        fit_label_encoders_on_train,
        assert_no_overlap_between_train_and_test,
        round_columns,
        prepare_train_val_numeric_splits,
    )


def test_round_columns_one_decimal() -> None:
    # WHY: Ensure rounding is applied only to provided columns and is robust to missing ones.
    df = pd.DataFrame({
        "fat-percentage": [12.345, 18.999, None],
        "bmi-calc": [21.2345, 27.5555, 30.0],
        "protein-per-kg": [0.876, 1.234, 1.999],
        "burns-calories-(per-30-min)-bc": [123.44, 55.51, 0.0],
        "other": [1.2345, 2.3456, 3.4567],
    })
    cols = [
        "fat-percentage",
        "bmi-calc",
        "protein-per-kg",
        "burns-calories-(per-30-min)-bc",
        "cal-balance",  # missing → ignored
    ]
    out = round_columns(df, cols, decimals=1)
    assert out["fat-percentage"].tolist() == [12.3, 19.0, None]
    assert out["bmi-calc"].tolist() == [21.2, 27.6, 30.0]
    assert out["protein-per-kg"].tolist() == [0.9, 1.2, 2.0]
    assert out["burns-calories-(per-30-min)-bc"].tolist() == [123.4, 55.5, 0.0]
    # Unspecified column must be untouched
    assert out["other"].equals(df["other"])  # type: ignore[no-untyped-call]


def test_read_best_params_returns_best_dict(tmp_path: Path) -> None:
    params = {"best_params": {"n_estimators": 64, "max_depth": 8}}
    json_path = tmp_path / "best_params.json"
    json_path.write_text(json.dumps(params), encoding="utf-8")

    result = read_best_params(json_path)

    assert result == params["best_params"]


def test_read_best_params_missing_key_raises(tmp_path: Path) -> None:
    json_path = tmp_path / "invalid.json"
    json_path.write_text(json.dumps({"wrong": {}}), encoding="utf-8")

    with pytest.raises(KeyError):
        read_best_params(json_path)


def test_save_training_results_persists_payload(tmp_path: Path) -> None:
    payload = {"metric": 1.23, "params": {"max_depth": 6}}
    target = tmp_path / "metrics" / "rf_summary.json"

    saved_path = save_training_results(payload, target)

    assert saved_path.exists()
    with saved_path.open("r", encoding="utf-8") as handle:
        content = json.load(handle)
    assert content == payload


def test_fit_label_encoders_on_train_maps_unseen_to_negative_one() -> None:
    train_df = pd.DataFrame({
        "color": ["red", "blue", "red", None],
        "city": ["Paris", "Lyon", None, "Paris"],
        "numeric": [1.0, 2.0, 3.0, 4.0],
    })
    val_df = pd.DataFrame({
        "color": ["red", "green", None],
        "city": ["Paris", "Berlin", "Lyon"],
        "numeric": [5.0, 6.0, 7.0],
    })

    (encoded_train, encoded_val), encoders = fit_label_encoders_on_train(
        train_df,
        [val_df],
    )

    assert list(encoders.keys()) == ["color", "city"]
    assert encoded_train["numeric"].equals(train_df["numeric"])
    assert set(encoded_train["color"]) == {0, 1, 2}
    assert -1 in encoded_val["color"].values
    assert -1 in encoded_val["city"].values


def test_prepare_train_val_numeric_splits_returns_numeric(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.RandomState(42)
    train_df = pd.DataFrame({
        "color": ["red", "blue", "green", "red"],
        "city": ["Paris", "Lyon", "Paris", "Nice"],
        "f1": rng.normal(size=4),
    })
    val_df = pd.DataFrame({
        "color": ["red", "yellow", "blue"],
        "city": ["Paris", "Berlin", "Lyon"],
        "f1": rng.normal(size=3),
    })
    test_df = pd.DataFrame({
        "color": ["green", "yellow"],
        "city": ["Nice", "Paris"],
        "f1": rng.normal(size=2),
    })

    def attach_target(df: pd.DataFrame) -> pd.DataFrame:
        base = df.copy()
        base["target"] = np.arange(len(base), dtype=float)
        return base

    train_df_with_target = attach_target(train_df)
    val_df_with_target = attach_target(val_df)
    test_df_with_target = attach_target(test_df)

    def fake_loader(_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return train_df_with_target, val_df_with_target, test_df_with_target

    monkeypatch.setattr("src.utils.load_splits_from_parquet", fake_loader)

    (X_train, y_train), (X_val, y_val) = prepare_train_val_numeric_splits(
        Path("unused.parquet"),
        "target",
    )

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_train.dtypes)
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_val.dtypes)
    assert -1 in X_val.select_dtypes(include=["int64", "int32"]).to_numpy()


def test_assert_no_overlap_between_train_and_test_with_id_columns() -> None:
    train = pd.DataFrame({"person_id": [1, 2, 3], "x": [10, 20, 30]})
    test_ok = pd.DataFrame({"person_id": [4, 5], "x": [40, 50]})
    test_ko = pd.DataFrame({"person_id": [3, 6], "x": [60, 70]})

    # No overlap → no exception
    assert_no_overlap_between_train_and_test(train, test_ok, id_columns=["person_id"])  # type: ignore[name-defined]

    # Overlap on id → raises
    with pytest.raises(ValueError):
        assert_no_overlap_between_train_and_test(train, test_ko, id_columns=["person_id"])  # type: ignore[name-defined]


def test_assert_no_overlap_between_train_and_test_fallback_full_row() -> None:
    # Full-row duplicate across splits triggers error when id_columns not provided
    row = {"gender": "Male", "age": 30, "weight-(kg)": 70.0}
    train = pd.DataFrame([row, {**row, "age": 31}])
    test_ok = pd.DataFrame([{**row, "age": 32}])  # different row → okay
    test_ko = pd.DataFrame([row])  # identical row → overlap

    assert_no_overlap_between_train_and_test(train, test_ok)  # type: ignore[name-defined]

    with pytest.raises(ValueError):
        assert_no_overlap_between_train_and_test(train, test_ko)  # type: ignore[name-defined]


def test_get_logger_configures_single_stdout_handler() -> None:
    # WHY: Ensure centralized logging sets a single stdout handler and avoids handler duplication.
    import src.utils as utils

    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(logging.WARNING)

    utils = importlib.reload(utils)

    first = utils.get_logger("sample")
    second = utils.get_logger("other")

    handlers = logging.getLogger().handlers
    assert len(handlers) == 1
    handler = handlers[0]
    assert getattr(handler, "stream", None) is sys.stdout
    assert first.name == "sample"
    assert second.name == "other"

    utils.get_logger("sample")
    assert len(logging.getLogger().handlers) == 1



if __name__ == "__main__":  # pragma: no cover
    pytest.main(["-vv", __file__])
