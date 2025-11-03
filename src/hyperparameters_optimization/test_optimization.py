from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from src.hyperparameters_optimization.optimization import (
    optimize_lightgbm_hyperparameters,
    optimize_random_forest_hyperparameters,
)


def _make_mock_split(
    rng: np.random.RandomState,
    n: int,
    *,
    coeff_f1: float = 2.0,
    coeff_f2: float = -0.5,
    noise_scale: float = 0.1,
) -> pd.DataFrame:
    """Create a synthetic dataset split with a controlled signal-to-noise ratio.

    WHY: Ensure deterministic and reproducible data for unit tests without real files.

    Args:
        rng: Random state used to draw samples deterministically.
        n: Number of rows to generate for the split.
        coeff_f1: Weight applied to feature ``f1`` when generating the target.
        coeff_f2: Weight applied to feature ``f2`` when generating the target.
        noise_scale: Standard deviation of Gaussian noise added to the target.

    Returns:
        Mocked split containing feature columns and a target column.
    """
    X = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    })
    y = (
        coeff_f1 * X["f1"]
        + coeff_f2 * X["f2"]
        + rng.normal(scale=noise_scale, size=n)
    )
    df = pd.concat([X, pd.Series(y, name="target")], axis=1)
    return df

def test_optimize_lightgbm_hyperparameters_with_mocked_splits(
    monkeypatch: pytest.MonkeyPatch,
    random_state_factory: Callable[[int | None], np.random.RandomState],
) -> None:
    """Validate that optimization runs end-to-end against mocked dataset splits.

    WHY: Guarantee the search logic remains functional when reading pre-split tabular data.
    """
    rng = random_state_factory(0)
    train_df = _make_mock_split(rng, 60)
    val_df = _make_mock_split(rng, 20)
    test_df = _make_mock_split(rng, 10)

    def fake_loader(_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return the mocked dataset splits regardless of input path.

        WHY: Decouple the test from filesystem dependencies to keep it hermetic.
        """
        return train_df, val_df, test_df

    # Mock dataset loading to avoid real files
    monkeypatch.setattr("src.utils.load_splits_from_parquet", fake_loader)

    best_params, best_value, val_summary = optimize_lightgbm_hyperparameters(
        parquet_path=Path("/does/not/matter.parquet"),
        target_column="target",
        n_trials=5,
        random_state=123,
    )

    assert isinstance(best_params, dict)
    assert isinstance(best_value, float)
    assert isinstance(val_summary, dict)
    assert "val_mse" in val_summary

    # Basic sanity checks on explored space keys (LightGBM)
    for key in [
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    ]:
        assert key in best_params


def test_optimize_lightgbm_handles_categorical_unseen_values(
    monkeypatch: pytest.MonkeyPatch,
    random_state_factory: Callable[[int | None], np.random.RandomState],
) -> None:
    rng = random_state_factory(1)

    train_df = pd.DataFrame({
        "color": [
            "red",
            "blue",
            "yellow",
            "red",
            "blue",
            "yellow",
            "red",
            "blue",
            "yellow",
        ],
        "city": [
            "Paris",
            "Lyon",
            "Marseille",
            "Paris",
            "Lyon",
            "Marseille",
            "Paris",
            "Lyon",
            "Marseille",
        ],
        "f1": rng.normal(size=9),
    })

    val_df = pd.DataFrame({
        "color": ["red", "green", "blue", "green", "blue", "red"],
        "city": ["Paris", "Nice", "Lyon", "Berlin", "Nice", "Paris"],
        "f1": rng.normal(size=6),
    })

    test_df = pd.DataFrame({
        "color": ["yellow", "green", "red"],
        "city": ["Nice", "Berlin", "Paris"],
        "f1": rng.normal(size=3),
    })

    color_weight = {
        "red": 1.0,
        "blue": 0.6,
        "yellow": 0.8,
        "green": 0.2,
    }
    city_weight = {
        "Paris": 0.3,
        "Lyon": 0.1,
        "Marseille": 0.5,
        "Nice": 0.2,
        "Berlin": 0.7,
    }

    def _attach_target(df: pd.DataFrame) -> pd.DataFrame:
        base = df.copy()
        noise = rng.normal(scale=0.01, size=len(base))
        target_values: List[float] = []
        for idx, row in enumerate(base.itertuples(index=False)):
            target_values.append(
                color_weight[getattr(row, "color")]
                + city_weight[getattr(row, "city")]
                + 0.1 * float(getattr(row, "f1"))
                + float(noise[idx])
            )
        base["target"] = target_values
        return base

    train_df = _attach_target(train_df)
    val_df = _attach_target(val_df)
    test_df = _attach_target(test_df)

    def fake_loader(_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return train_df, val_df, test_df

    monkeypatch.setattr("src.utils.load_splits_from_parquet", fake_loader)

    captured_val_inputs: list[pd.DataFrame] = []

    # Replace LightGBM factory with a dummy model compatible with LGBM.fit signature
    class DummyLGBM:
        def __init__(self) -> None:
            self.params: dict[str, object] = {}
            self.best_iteration_ = None

        def get_params(self) -> dict[str, object]:
            return {}

        def set_params(self, **params: object) -> "DummyLGBM":
            self.params.update(params)
            return self

        def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: object) -> "DummyLGBM":
            return self

        def predict(self, X: pd.DataFrame, **kwargs: object) -> np.ndarray:  # type: ignore[name-defined]
            captured_val_inputs.append(X.copy())
            return np.zeros(len(X))

    dummy_factory = lambda random_state: DummyLGBM()
    monkeypatch.setattr("src.models.models.create_lightgbm_regressor", dummy_factory)
    monkeypatch.setattr(
        "src.hyperparameters_optimization.optimization.create_lightgbm_regressor",
        dummy_factory,
    )

    optimize_lightgbm_hyperparameters(
        parquet_path=Path("/tmp/raw_splits.parquet"),
        target_column="target",
        n_trials=2,
        random_state=123,
    )

    assert captured_val_inputs, "Expected validation inputs captured from predict calls."
    assert any(-1 in df["color"].values for df in captured_val_inputs)
    assert any(-1 in df["city"].values for df in captured_val_inputs)


def test_optimize_random_forest_hyperparameters_with_mocked_splits(
    monkeypatch: pytest.MonkeyPatch,
    random_state_factory: Callable[[int | None], np.random.RandomState],
) -> None:
    rng = random_state_factory(7)
    train_df = _make_mock_split(rng, 50, coeff_f1=1.5, coeff_f2=-0.25, noise_scale=0.05)
    val_df = _make_mock_split(rng, 18, coeff_f1=1.5, coeff_f2=-0.25, noise_scale=0.05)
    test_df = _make_mock_split(rng, 10, coeff_f1=1.5, coeff_f2=-0.25, noise_scale=0.05)

    def fake_loader(_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return train_df, val_df, test_df

    monkeypatch.setattr("src.utils.load_splits_from_parquet", fake_loader)

    best_params, best_value, val_summary = optimize_random_forest_hyperparameters(
        parquet_path=Path("/unused.parquet"),
        target_column="target",
        n_trials=3,
        random_state=123,
    )

    assert isinstance(best_params, dict)
    assert isinstance(best_value, float)
    assert isinstance(val_summary, dict)
    assert "val_mse" in val_summary
    for key in [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "bootstrap",
    ]:
        assert key in best_params


def test_optimize_random_forest_handles_categorical(
    monkeypatch: pytest.MonkeyPatch,
    random_state_factory: Callable[[int | None], np.random.RandomState],
) -> None:
    rng = random_state_factory(11)

    train_df = pd.DataFrame({
        "color": ["red", "blue", "green", "red", "blue", "green"],
        "city": ["Paris", "Lyon", "Paris", "Lyon", "Paris", "Lyon"],
        "f": rng.normal(size=6),
    })
    val_df = pd.DataFrame({
        "color": ["red", "yellow", "blue"],
        "city": ["Paris", "Nice", "Paris"],
        "f": rng.normal(size=3),
    })
    test_df = pd.DataFrame({
        "color": ["green", "yellow"],
        "city": ["Nice", "Paris"],
        "f": rng.normal(size=2),
    })

    def attach_target(df: pd.DataFrame) -> pd.DataFrame:
        base = df.copy()
        weight = {"red": 1.0, "blue": 0.7, "green": 0.5, "yellow": 0.2}
        city_w = {"Paris": 0.3, "Lyon": 0.1, "Nice": 0.2}
        base["target"] = [
            weight.get(color, 0.0) + city_w.get(city, 0.0) + 0.1
            for color, city in zip(base["color"], base["city"])
        ]
        return base

    train_df = attach_target(train_df)
    val_df = attach_target(val_df)
    test_df = attach_target(test_df)

    monkeypatch.setattr("src.utils.load_splits_from_parquet", lambda _p: (train_df, val_df, test_df))

    best_params, best_value, val_summary = optimize_random_forest_hyperparameters(
        parquet_path=Path("/unused.parquet"),
        target_column="target",
        n_trials=1,
        random_state=123,
    )

    assert isinstance(best_params, dict)
    assert isinstance(best_value, float)
    assert "val_mse" in val_summary


def _mock_cli_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(123)
    train = pd.DataFrame({
        "Weight (kg)": rng.normal(loc=70, scale=9, size=20),
        "x": rng.randn(20),
    })
    val = pd.DataFrame({
        "Weight (kg)": rng.normal(loc=69, scale=9, size=8),
        "x": rng.randn(8),
    })
    test = pd.DataFrame({
        "Weight (kg)": rng.normal(loc=68, scale=9, size=6),
        "x": rng.randn(6),
    })
    return train, val, test


def _setup_cli_mocks(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, int], list[dict[str, object]]]:
    monkeypatch.setattr(
        "src.hyperparameters_optimization.main.load_splits_from_parquet",
        lambda _p: _mock_cli_splits(),
    )

    saved_payloads: list[dict[str, object]] = []

    def _fake_save(payload: dict[str, object], path: str) -> str:
        saved_payloads.append(payload)
        return path

    monkeypatch.setattr(
        "src.hyperparameters_optimization.main.save_training_results",
        _fake_save,
    )

    calls = {"lgbm": 0, "rf": 0}

    def _fake_lgbm(**_kwargs: object) -> tuple[dict[str, object], float, dict[str, float]]:
        calls["lgbm"] += 1
        return {"n_estimators": 10}, 1.0, {"val_mse": 1.0}

    def _fake_rf(**_kwargs: object) -> tuple[dict[str, object], float, dict[str, float]]:
        calls["rf"] += 1
        return {"n_estimators": 20}, 1.1, {"val_mse": 1.1}

    monkeypatch.setattr(
        "src.hyperparameters_optimization.main.optimize_lightgbm_hyperparameters",
        _fake_lgbm,
    )
    monkeypatch.setattr(
        "src.hyperparameters_optimization.main.optimize_random_forest_hyperparameters",
        _fake_rf,
    )

    return calls, saved_payloads


def test_cli_default_runs_both(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.hyperparameters_optimization.main import main as cli_main

    calls, saved = _setup_cli_mocks(monkeypatch)

    monkeypatch.setattr(sys, "argv", ["prog"])
    cli_main()

    assert calls["lgbm"] == 1
    assert calls["rf"] == 1
    assert any("val_metrics" in payload for payload in saved)


def test_cli_runs_only_lightgbm(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.hyperparameters_optimization.main import main as cli_main

    calls, _ = _setup_cli_mocks(monkeypatch)

    monkeypatch.setattr(sys, "argv", ["prog", "--models", "lightgbm"])
    cli_main()

    assert calls["lgbm"] == 1
    assert calls["rf"] == 0


def test_cli_runs_only_random_forest(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.hyperparameters_optimization.main import main as cli_main

    calls, _ = _setup_cli_mocks(monkeypatch)

    monkeypatch.setattr(sys, "argv", ["prog", "--models", "random_forest"])
    cli_main()

    assert calls["lgbm"] == 0
    assert calls["rf"] == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
