"""Tests for evaluation helpers and CLI entry point."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

try:
    import shap as _shap_check  # noqa: F401  # pragma: no cover - import guard
except Exception as exc:  # pragma: no cover - environment dependency
    pytest.skip(f"shap import failed: {exc}", allow_module_level=True)

from src.constants import (
    DEFAULT_RANDOM_STATE,
    LIGHTGBM_SHAP_DIR,
    RANDOM_FOREST_SHAP_DIR,
    RELATIVE_SHAP_PLOTS_DIR,
    TARGET_COLUMN,
)
from src.eval.eval import evaluate_lightgbm_on_test


def _make_linear_split(
    rng: np.random.RandomState, n: int, *, coef_f1: float = 3.0, coef_f2: float = 2.0
) -> pd.DataFrame:
    """Return deterministic linear synthetic data to reason about metrics."""

    features = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
        }
    )
    targets = coef_f1 * features["f1"] + coef_f2 * features["f2"]
    return pd.concat([features, pd.Series(targets, name=TARGET_COLUMN)], axis=1)


def test_evaluate_lightgbm_on_test_perfect_predictions(
    monkeypatch: pytest.MonkeyPatch,
    random_state_factory: Callable[[int | None], np.random.RandomState],
) -> None:
    """evaluation metrics are exact when the model reproduces the generator."""

    rng = random_state_factory(123)
    train_df = _make_linear_split(rng, 20)
    val_df = _make_linear_split(rng, 10)
    test_df = _make_linear_split(rng, 12)

    monkeypatch.setattr(
        "src.eval.eval.load_splits_from_parquet",
        lambda _path: (train_df, val_df, test_df),
    )

    class _PerfectModel:
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return 3.0 * frame["f1"].to_numpy() + 2.0 * frame["f2"].to_numpy()

    monkeypatch.setattr("src.eval.eval.joblib.load", lambda _p: _PerfectModel())

    metrics = evaluate_lightgbm_on_test(
        parquet_path=Path("/does/not/matter.parquet"),
        model_path=Path("/does/not/matter.joblib"),
        target_column=TARGET_COLUMN,
    )

    assert isinstance(metrics, dict)
    assert pytest.approx(metrics["mae"], abs=1e-12) == 0.0
    assert pytest.approx(metrics["mse"], abs=1e-12) == 0.0
    assert pytest.approx(metrics["rmse"], abs=1e-12) == 0.0
    assert pytest.approx(metrics["r2"], abs=1e-12) == 1.0
    assert pytest.approx(metrics["mape"], abs=1e-12) == 0.0
    assert pytest.approx(metrics["median_ae"], abs=1e-12) == 0.0
    assert pytest.approx(metrics["explained_variance"], abs=1e-12) == 1.0


def test_cli_eval_lightgbm_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CLI prints LightGBM metrics and saved path for single-model evaluation."""

    from src.eval.main import main as eval_cli_main

    eval_dir = tmp_path / "results" / "eval"
    metrics_payload = {
        "mae": 0.1,
        "mse": 0.01,
        "rmse": 0.1,
        "r2": 0.9,
        "mape": 0.05,
        "median_ae": 0.08,
        "explained_variance": 0.88,
        "y_std": 1.0,
        "pred_std": 0.95,
        "residual_std": 0.2,
    }
    expected_shap_dir = LIGHTGBM_SHAP_DIR
    shap_plot_relative = str(RELATIVE_SHAP_PLOTS_DIR / "LightGBM" / "lightgbm_shap_beeswarm.png")
    shap_payload = {
        "plot_path": str(expected_shap_dir / "lightgbm_shap_beeswarm.png"),
        "expected_value": 0.0,
        "feature_impacts": [],
    }
    expected_saved_shap = {**shap_payload, "plot_path": shap_plot_relative}

    def fake_eval(
        *,
        parquet_path: Path,
        model_path: Path,
        target_column: str,
        shap_output_dir: Path,
        shap_max_display: int,
        model_label: str,
        shap_sample_size: int | None,
        shap_random_state: int,
    ) -> dict[str, Any]:
        assert parquet_path == Path("data/dataset_splits_encoded.parquet")
        assert model_path == Path("results/models/lightgbm.joblib")
        assert target_column == TARGET_COLUMN
        assert shap_output_dir == expected_shap_dir
        assert shap_max_display == 20
        assert model_label == "lightgbm"
        assert shap_sample_size is None
        assert shap_random_state == DEFAULT_RANDOM_STATE
        return {**metrics_payload, "shap": shap_payload}

    def fake_save(payload: dict[str, Any], json_path: Path) -> Path:
        # WHY: Persist to tmp_path to emulate real behaviour without touching repo state.
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        return json_path.resolve()

    monkeypatch.setattr("src.eval.main.evaluate_lightgbm_on_test", fake_eval)
    monkeypatch.setattr("src.eval.main.save_training_results", fake_save)

    caplog.set_level(logging.INFO, logger="src.eval.main")

    rc = eval_cli_main(
        [
            "--parquet",
            "data/dataset_splits_encoded.parquet",
            "--model",
            "results/models/lightgbm.joblib",
            "--models",
            "lightgbm",
            "--target-column",
            TARGET_COLUMN,
            "--eval-dir",
            str(eval_dir),
        ]
    )

    assert rc == 0
    messages = caplog.messages
    # WHY: Ensure CLI still surfaces evaluation metrics and persisted artifact via logger output.
    assert messages, "Expected logging output from evaluation CLI"
    assert any(
        "LightGBM metrics" in message and json.dumps(metrics_payload) in message
        for message in messages
    )
    assert any("lightgbm_test_metrics.json" in message for message in messages)
    saved_report = json.loads((eval_dir / "lightgbm_test_metrics.json").read_text(encoding="utf-8"))
    assert saved_report["metrics"] == metrics_payload
    assert saved_report["shap"] == expected_saved_shap


def test_cli_eval_with_random_forest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """CLI persists summaries for both models when artifacts exist."""

    from src.eval.main import main as eval_cli_main

    lgbm_metrics = {
        "mae": 0.1,
        "mse": 0.01,
        "rmse": 0.1,
        "r2": 0.9,
        "mape": 0.05,
        "median_ae": 0.08,
        "explained_variance": 0.88,
        "y_std": 1.1,
        "pred_std": 0.9,
        "residual_std": 0.25,
    }
    rf_metrics = {
        "mae": 0.2,
        "mse": 0.02,
        "rmse": 0.141421356,
        "r2": 0.85,
        "mape": 0.06,
        "median_ae": 0.1,
        "explained_variance": 0.8,
        "y_std": 1.2,
        "pred_std": 1.0,
        "residual_std": 0.3,
    }

    def fake_eval(
        *,
        parquet_path: Path,
        model_path: Path,
        target_column: str,
        shap_output_dir: Path,
        shap_max_display: int,
        model_label: str,
        shap_sample_size: int | None,
        shap_random_state: int,
    ) -> dict[str, Any]:
        expected_dir = LIGHTGBM_SHAP_DIR if model_label == "lightgbm" else RANDOM_FOREST_SHAP_DIR
        assert shap_output_dir == expected_dir
        if model_label == "random_forest":
            assert shap_max_display == 0
            assert shap_sample_size is None
        else:
            assert shap_max_display == 20
            assert shap_sample_size is None
        assert shap_random_state == DEFAULT_RANDOM_STATE
        shap_stub = {
            "plot_path": str(expected_dir / f"{model_label}_shap_beeswarm.png"),
            "expected_value": 1.0,
            "feature_impacts": [
                {
                    "feature": "f1",
                    "max_positive": 0.2,
                    "max_negative": -0.3,
                    "positive_rate": 0.7,
                    "negative_rate": 0.3,
                    "max_intensity": 0.3,
                }
            ],
        }
        if model_path.name.endswith("random_forest.joblib"):
            return {**rf_metrics, "shap": None}
        return {**lgbm_metrics, "shap": shap_stub}

    def fake_save(payload: dict[str, Any], json_path: Path) -> Path:
        # WHY: Mirror production behaviour so CLI can re-emit exact path strings.
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        return json_path

    monkeypatch.setattr("src.eval.main.evaluate_lightgbm_on_test", fake_eval)
    monkeypatch.setattr("src.eval.main.save_training_results", fake_save)

    rf_artifact = tmp_path / "models" / "random_forest.joblib"
    rf_artifact.parent.mkdir(parents=True, exist_ok=True)
    rf_artifact.write_bytes(b"J")

    eval_dir = tmp_path / "results" / "eval"

    rc = eval_cli_main(
        [
            "--parquet",
            "data/dataset_splits_encoded.parquet",
            "--model",
            "results/models/lightgbm.joblib",
            "--rf-model",
            str(rf_artifact),
            "--eval-dir",
            str(eval_dir),
        ]
    )

    assert rc == 0
    lgbm_json = json.loads((eval_dir / "lightgbm_test_metrics.json").read_text(encoding="utf-8"))
    rf_json = json.loads((eval_dir / "random_forest_test_metrics.json").read_text(encoding="utf-8"))
    assert lgbm_json["metrics"] == lgbm_metrics
    assert rf_json["metrics"] == rf_metrics
    expected_lgbm_shap = {
        "plot_path": str(RELATIVE_SHAP_PLOTS_DIR / "LightGBM" / "lightgbm_shap_beeswarm.png"),
        "expected_value": 1.0,
        "feature_impacts": [
            {
                "feature": "f1",
                "max_positive": 0.2,
                "max_negative": -0.3,
                "positive_rate": 0.7,
                "negative_rate": 0.3,
                "max_intensity": 0.3,
            }
        ],
    }
    assert lgbm_json["shap"] == expected_lgbm_shap
    assert rf_json["shap"] is None


def test_evaluate_lightgbm_on_test_with_shap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    random_state_factory: Callable[[int | None], np.random.RandomState],
) -> None:
    """SHAP report contains signed impact stats and plot artefact."""

    rng = random_state_factory(7)
    train_df = _make_linear_split(rng, 20)
    val_df = _make_linear_split(rng, 10)
    test_df = _make_linear_split(rng, 8)

    shap_dir = tmp_path / "reports" / "shap"

    monkeypatch.setattr(
        "src.eval.eval.load_splits_from_parquet",
        lambda _path: (train_df, val_df, test_df),
    )

    class _LinearModel:
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return 3.0 * frame["f1"].to_numpy() + 2.0 * frame["f2"].to_numpy()

    monkeypatch.setattr("src.eval.eval.joblib.load", lambda _p: _LinearModel())

    class _Explainer:
        def __init__(self, _model: Any) -> None:
            self._model = _model

        def __call__(self, features: pd.DataFrame) -> Any:
            values = np.tile(np.array([[0.4, -0.5]]), (len(features), 1))
            return SimpleNamespace(
                values=values,
                base_values=np.ones(len(features)),
            )

    monkeypatch.setattr("src.eval.eval.shap.TreeExplainer", _Explainer)

    saved_paths: list[Path] = []

    def _fake_beeswarm(*_args: Any, **_kwargs: Any) -> None:
        return None

    def _fake_savefig(path: Path, **_kwargs: Any) -> None:
        saved_paths.append(Path(path))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("plot", encoding="utf-8")

    monkeypatch.setattr("src.eval.eval.shap.plots.beeswarm", _fake_beeswarm)
    monkeypatch.setattr("src.eval.eval.plt.savefig", _fake_savefig)

    metrics = evaluate_lightgbm_on_test(
        parquet_path=Path("/dummy.parquet"),
        model_path=Path("/dummy.joblib"),
        target_column=TARGET_COLUMN,
        shap_output_dir=shap_dir,
        shap_max_display=3,
        model_label="lightgbm",
    )

    shap_payload = metrics["shap"]
    assert shap_payload is not None
    assert Path(shap_payload["plot_path"]) in saved_paths
    assert shap_payload["expected_value"] == 1.0
    assert shap_payload["feature_impacts"], "Feature impacts should not be empty"
    first_feature = shap_payload["feature_impacts"][0]
    assert set(first_feature) == {
        "feature",
        "max_positive",
        "max_negative",
        "positive_rate",
        "negative_rate",
        "max_intensity",
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
