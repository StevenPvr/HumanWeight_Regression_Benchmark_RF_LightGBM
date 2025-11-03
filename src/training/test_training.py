from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from src.training.training import train_lightgbm_with_best_params

from src.constants import DEFAULT_RANDOM_STATE, TARGET_COLUMN
from src.utils import split_features_target, ensure_numeric_columns


def _make_mock_split(
    rng: np.random.RandomState,
    n: int,
    *,
    coef_f1: float = 2.0,
    coef_f2: float = -0.5,
    noise: float = 0.1,
) -> pd.DataFrame:
    """Return synthetic regression data with configurable signal strength."""

    X = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    })
    signal = coef_f1 * X["f1"] + coef_f2 * X["f2"]
    if noise > 0.0:
        # WHY: Allow tests to toggle stochastic noise for deterministic assertions when needed.
        signal = signal + rng.normal(scale=noise, size=n)
    y = signal
    return pd.concat([X, pd.Series(y, name=TARGET_COLUMN)], axis=1)


def test_train_random_forest_with_best_params_monkeypatched(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    random_state_factory: Callable[[int | None], np.random.RandomState],
) -> None:
    rng = random_state_factory(0)
    train_df = _make_mock_split(rng, 80)
    val_df = _make_mock_split(rng, 20)
    test_df = _make_mock_split(rng, 10)

    def fake_loader(_path: Path):
        return train_df, val_df, test_df

    # Mock dataset loading to avoid filesystem dependency
    # Patch the symbol as imported into the module under test
    monkeypatch.setattr("src.training.training.load_splits_from_parquet", fake_loader)

    from src.training import training as training_module

    original_split = training_module.split_features_target
    recorded_lengths: list[int] = []

    def tracking_split(df: pd.DataFrame, target_column: str):
        recorded_lengths.append(len(df))
        return original_split(df, target_column)

    monkeypatch.setattr(
        "src.training.training.split_features_target",
        tracking_split,
    )

    # Create a temporary JSON with best params
    params = {
        "best_params": {
            "n_estimators": 64,
            "max_depth": 6,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": 0.8,
            "bootstrap": True,
        }
    }
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "best_lightgbm_params.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f)

    model = train_lightgbm_with_best_params(
        parquet_path=Path("/does/not/matter.parquet"),
        params_json_path=json_path,
        target_column=TARGET_COLUMN,
        random_state=DEFAULT_RANDOM_STATE,
    )

    assert isinstance(model, LGBMRegressor)
    # Expect two calls: first on train, then on val for monitoring
    assert recorded_lengths[:2] == [len(train_df), len(val_df)]
    assert getattr(model, "n_features_in_", None) == 2
    # Verify params were applied
    # n_estimators parameter should reflect tuned cap
    assert 0 < model.get_params()["n_estimators"] <= 64
    assert model.get_params()["max_depth"] == 6

    # Basic prediction sanity on validation split
    X_val_raw, _y_val = split_features_target(val_df, TARGET_COLUMN)
    X_val = ensure_numeric_columns(X_val_raw)
    preds = model.predict(X_val)
    # Normalize to ndarray to satisfy strict type checkers (predict may return sparse/list)
    assert np.asarray(preds).shape == (len(val_df),)


def test_save_random_forest_model_monkeypatched(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    random_state_factory: Callable[[int | None], np.random.RandomState],
) -> None:
    rng = random_state_factory(1)
    train_df = _make_mock_split(rng, 60)
    val_df = _make_mock_split(rng, 20)
    test_df = _make_mock_split(rng, 10)

    def fake_loader(_path: Path):
        return train_df, val_df, test_df

    # Patch the symbol as imported into the module under test
    monkeypatch.setattr("src.training.training.load_splits_from_parquet", fake_loader)

    # Minimal best params JSON
    params = {
        "best_params": {
            "n_estimators": 32,
            "max_depth": 5,
        }
    }
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    params_path = results_dir / "best_lgbm_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f)

    # Train a model
    model = train_lightgbm_with_best_params(
        parquet_path=Path("/unused.parquet"),
        params_json_path=params_path,
        target_column=TARGET_COLUMN,
        random_state=DEFAULT_RANDOM_STATE,
    )

    # Save the model without extension to verify default .joblib behavior
    from src.training.training import save_lightgbm_model

    out_base = tmp_path / "model" / "rf_model"
    saved_path = save_lightgbm_model(model, str(out_base))

    assert saved_path.endswith(".joblib")
    assert os.path.exists(saved_path)

    # Reload and predict to ensure artifact is valid
    loaded = joblib.load(saved_path)
    assert isinstance(loaded, LGBMRegressor)

    X_val_raw, _y_val = split_features_target(val_df, TARGET_COLUMN)
    X_val = ensure_numeric_columns(X_val_raw)
    preds = loaded.predict(X_val)
    assert np.asarray(preds).shape == (len(val_df),)


def test_cli_main_trains_and_saves(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    random_state_factory: Callable[[int | None], np.random.RandomState],
) -> None:
    # Import here to ensure we patch the correct module symbols
    from src.training.main import main as cli_main

    called = {"train": False, "save": False, "summary": False}

    expected_parquet = Path("dummy.parquet")
    expected_params = tmp_path / "params.json"
    expected_target = TARGET_COLUMN
    expected_seed = DEFAULT_RANDOM_STATE
    out_base = tmp_path / "out" / "rf_model"
    expected_return_path = str(out_base.with_suffix(".joblib"))

    def fake_train(parquet_path: Path, params_json_path: Path, *, target_column: str, random_state: int):
        assert parquet_path == expected_parquet
        assert params_json_path == expected_params
        assert target_column == expected_target
        assert random_state == expected_seed
        called["train"] = True
        return object()  # placeholder model, will be ignored by fake save

    def fake_save(model, model_path: str) -> str:
        assert model is not None
        assert model_path == str(out_base)
        called["save"] = True
        return expected_return_path

    # Provide a tiny validation set to allow CLI to compute MSE without touching FS
    rng = random_state_factory(42)
    df_train = _make_mock_split(rng, 10)
    df_val = _make_mock_split(rng, 5)
    df_test = _make_mock_split(rng, 5)

    class _Model:
        def predict(self, X):
            # simple dummy predictions close to target linear relation
            return 2.0 * X["f1"].to_numpy() - 0.5 * X["f2"].to_numpy()

    def fake_train_return_model(*args, **kwargs):
        # Keep the assertions via fake_train, then return a model-like object
        fake_train(*args, **kwargs)
        return _Model()

    def fake_loader(_path: Path):
        return df_train, df_val, df_test

    def fake_read_params(path: Path):
        assert path == expected_params
        return {"n_estimators": 64}

    def fake_save_summary(payload: dict, json_path: Path) -> Path:
        expected_path = out_base.parent / "rf_model_metrics.json"
        assert json_path == expected_path
        assert payload["model_path"] == expected_return_path
        # Validation MSE should be computed from the final model
        assert isinstance(payload["validation"]["mse"], float)
        assert payload["validation"]["mse"] >= 0.0
        # Ensure no CV references are present in classic training summary
        assert "cv_metrics" not in payload
        assert payload.get("training_rows") == len(df_train)
        assert payload.get("validation_rows") == len(df_val)
        called["summary"] = True
        return json_path

    # Patch functions used inside CLI
    monkeypatch.setattr("src.training.main.train_lightgbm_with_best_params", fake_train_return_model)
    monkeypatch.setattr("src.training.main.save_lightgbm_model", fake_save)
    monkeypatch.setattr("src.training.main.load_splits_from_parquet", lambda p: (df_train, df_val, df_test))
    monkeypatch.setattr("src.training.main.read_best_params", fake_read_params)
    monkeypatch.setattr("src.training.main.save_training_results", fake_save_summary)
    # SHAP usage removed from training CLI; no patch needed

    caplog.set_level(logging.INFO, logger="src.training.main")

    rc = cli_main([
        "--parquet", str(expected_parquet),
        "--params", str(expected_params),
        "--out", str(out_base),
        "--target-column", expected_target,
        "--random-state", str(expected_seed),
        "--no-random-forest",
    ])

    assert rc == 0
    messages = caplog.messages
    # WHY: Validate user-facing confirmation still mentions persisted artifact location.
    assert any(expected_return_path in message for message in messages)
    assert called["train"] and called["save"] and called["summary"]


def test_cli_main_uses_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    random_state_factory: Callable[[int | None], np.random.RandomState],
) -> None:
    from src.training import main as main_module
    from src.training.main import main as cli_main

    default_parquet = tmp_path / "splits.parquet"
    default_params = tmp_path / "best_params.json"
    default_out = tmp_path / "artifacts" / "random_forest"

    monkeypatch.setattr(main_module, "DEFAULT_PARQUET", default_parquet)
    monkeypatch.setattr(main_module, "DEFAULT_PARAMS", default_params)
    monkeypatch.setattr(main_module, "DEFAULT_MODEL_OUT", default_out)

    called = {"train": False, "save": False, "summary": False}

    class _Model:
        def predict(self, X):
            return 2.0 * X["f1"].to_numpy() - 0.5 * X["f2"].to_numpy()

    rng = random_state_factory(0)
    df_train = _make_mock_split(rng, 12)
    df_val = _make_mock_split(rng, 6)
    df_test = _make_mock_split(rng, 6)

    def fake_loader(path: Path):
        assert path == default_parquet
        return df_train, df_val, df_test

    def fake_train(parquet_path: Path, params_json_path: Path, *, target_column: str, random_state: int):
        assert parquet_path == default_parquet
        assert params_json_path == default_params
        assert target_column == TARGET_COLUMN
        assert random_state == DEFAULT_RANDOM_STATE
        called["train"] = True
        return _Model()

    def fake_save(model, model_path: str) -> str:
        assert isinstance(model, _Model)
        assert Path(model_path) == default_out
        called["save"] = True
        return str(default_out.with_suffix(".joblib"))

    def fake_read_params(path: Path):
        assert path == default_params
        return {"n_estimators": 32}

    def fake_save_summary(payload: dict, json_path: Path) -> Path:
        expected_path = default_out.parent / "lightgbm_metrics.json"
        assert json_path == expected_path
        assert payload["model_path"] == str(default_out.with_suffix(".joblib"))
        assert isinstance(payload["validation"]["mse"], float)
        assert payload["validation"]["mse"] >= 0.0
        # Ensure no CV references are present in classic training summary
        assert "cv_metrics" not in payload
        assert payload.get("training_rows") == len(df_train)
        assert payload.get("validation_rows") == len(df_val)
        called["summary"] = True
        return json_path

    monkeypatch.setattr("src.training.main.load_splits_from_parquet", fake_loader)
    monkeypatch.setattr("src.training.main.train_lightgbm_with_best_params", fake_train)
    monkeypatch.setattr("src.training.main.save_lightgbm_model", fake_save)
    monkeypatch.setattr("src.training.main.read_best_params", fake_read_params)
    monkeypatch.setattr("src.training.main.save_training_results", fake_save_summary)

    caplog.set_level(logging.INFO, logger="src.training.main")

    rc = cli_main(["--no-random-forest"])

    assert rc == 0
    messages = caplog.messages
    # WHY: Ensure default workflow still reports artifact path through logger output.
    assert any(str(default_out.with_suffix(".joblib")) in message for message in messages)
    assert called["train"] and called["save"] and called["summary"]


if __name__ == "__main__":
    pytest.main([__file__])
