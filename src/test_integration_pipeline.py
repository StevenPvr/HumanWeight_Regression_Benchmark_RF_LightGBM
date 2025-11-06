from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


def _mock_df(n: int, rng: np.random.RandomState) -> pd.DataFrame:
    """Small synthetic dataset to flow through stages."""

    return pd.DataFrame(
        {
            "Weight (kg)": rng.normal(loc=70, scale=10, size=n),
            "physical exercise": rng.randint(0, 2, size=n),
            "gender": rng.choice(["M", "F"], size=n),
            "age": rng.randint(18, 70, size=n),
        }
    )


def test_stage_cleaning_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pandas_plot_noop: None,
    plt_stub: object,
    random_state_factory: Callable[[int | None], np.random.RandomState],
) -> None:
    from src.data_cleaning.main import main as cleaning_main

    rng = random_state_factory(42)
    raw_df = _mock_df(40, rng)
    monkeypatch.setattr("src.data_cleaning.main.pd.read_csv", lambda _p: raw_df.copy())
    monkeypatch.setattr("src.data_cleaning.main.plt", plt_stub)

    out_csv = tmp_path / "data" / "dataset_cleaned_final.csv"
    out_parquet = tmp_path / "data" / "dataset_cleaned_final.parquet"

    def _fake_save(
        dataset: pd.DataFrame,
        output_name: str = "dataset_cleaned_final",
        output_dir: Path | str | None = None,
    ) -> None:
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)
        dataset.to_csv(out_csv, index=False)
        dataset.to_parquet(out_parquet, index=False)

    monkeypatch.setattr("src.data_cleaning.main.save_cleaned_dataset", _fake_save)
    cleaning_main()
    assert out_csv.exists() and out_parquet.exists()


def test_stage_preparation_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    random_state_factory: Callable[[int | None], np.random.RandomState],
) -> None:
    from src.data_preparation.main import main as prep_main

    rng = random_state_factory(7)
    cleaned_df = _mock_df(40, rng)
    cleaned_df.columns = ["weight-(kg)", "physical-exercise", "gender", "age"]
    monkeypatch.setattr("src.data_preparation.main.pd.read_csv", lambda _p: cleaned_df.copy())

    out_csv = tmp_path / "data" / "dataset_splits_encoded.csv"
    out_parquet = tmp_path / "data" / "dataset_splits_encoded.parquet"

    def _save(
        train_df: pd.DataFrame, test_df: pd.DataFrame, csv_path: str, parquet_path: str
    ) -> None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.concat([train_df.assign(split="train"), test_df.assign(split="test")]).to_csv(
            out_csv, index=False
        )
        pd.concat([train_df.assign(split="train"), test_df.assign(split="test")]).to_parquet(
            out_parquet, index=False
        )

    monkeypatch.setattr("src.data_preparation.main.save_splits_with_marker", _save)
    monkeypatch.setattr(
        "src.data_preparation.main.save_label_encoders_mappings", lambda e, j, c: None
    )
    prep_main()
    assert out_csv.exists() and out_parquet.exists()


def test_stage_hpo_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    random_state_factory: Callable[[int | None], np.random.RandomState],
) -> None:
    from src.hyperparameters_optimization.main import main as hpo_main

    rng = random_state_factory(1)
    base = _mock_df(30, rng)
    base.columns = ["weight-(kg)", "physical-exercise", "gender", "age"]
    train_df = base.sample(20, random_state=1).reset_index(drop=True)
    test_df = base.drop(index=list(train_df.index)).reset_index(drop=True)

    monkeypatch.setattr("src.hyperparameters_optimization.main.RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(
        "src.hyperparameters_optimization.main.load_splits_from_parquet",
        lambda _p: (train_df, test_df),
    )

    def _opt_lgbm(*_a, **_k):
        return {"n_estimators": 64, "learning_rate": 0.1}, 1.0, {"val_mse": 1.0}

    def _opt_rf(*_a, **_k):
        return {"n_estimators": 50, "max_depth": 5}, 1.1, {"val_mse": 1.1}

    monkeypatch.setattr(
        "src.hyperparameters_optimization.main.optimize_lightgbm_hyperparameters",
        _opt_lgbm,
    )
    monkeypatch.setattr(
        "src.hyperparameters_optimization.main.optimize_random_forest_hyperparameters",
        _opt_rf,
    )
    monkeypatch.setattr(
        "src.hyperparameters_optimization.main.sys.argv",
        ["pytest", "--models", "both"],
    )
    hpo_main()


def test_stage_feature_engineering_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    random_state_factory: Callable[[int | None], np.random.RandomState],
) -> None:
    from src.feature_engineering.main import main as fe_main

    rng = random_state_factory(2)
    base = _mock_df(30, rng)
    base.columns = ["weight-(kg)", "physical-exercise", "gender", "age"]
    train_df = base.sample(20, random_state=2).reset_index(drop=True)
    test_df = base.drop(index=list(train_df.index)).reset_index(drop=True)

    monkeypatch.setattr(
        "src.feature_engineering.main.load_splits_from_parquet", lambda _p: (train_df, test_df)
    )
    importance_df = pd.DataFrame(
        {
            "feature": ["age", "gender"],
            "importance_mean": [0.1, 0.05],
            "importance_std": [0.01, 0.005],
        }
    )
    monkeypatch.setattr(
        "src.feature_engineering.main.compute_random_forest_permutation_importance",
        lambda *a, **k: importance_df,
    )

    out_json = tmp_path / "results" / "feature_importances.json"
    out_png = tmp_path / "plots" / "permutation" / "feature_importances.png"

    def _save_importances(df: pd.DataFrame, path: Path) -> None:
        os.makedirs(os.path.dirname(str(out_json)), exist_ok=True)
        Path(str(out_json)).write_text(json.dumps(df.to_dict(orient="records")), encoding="utf-8")

    def _plot_importances(
        df: pd.DataFrame, path: Path, top_n: int | None = None
    ) -> None:  # noqa: ARG001
        os.makedirs(os.path.dirname(str(out_png)), exist_ok=True)
        Path(str(out_png)).write_bytes(b"PNG")

    monkeypatch.setattr(
        "src.feature_engineering.main.save_feature_importances_to_json",
        _save_importances,
    )
    monkeypatch.setattr(
        "src.feature_engineering.main.plot_feature_importances",
        _plot_importances,
    )

    argv = [
        "pytest",
        "--input",
        str(tmp_path / "data" / "dataset_splits_encoded.parquet"),
        "--target",
        "weight-(kg)",
        "--output-json",
        str(out_json),
        "--output-plot",
        str(out_png),
    ]
    monkeypatch.setattr("src.feature_engineering.main.sys.argv", argv)
    fe_main()
    assert out_json.exists() and out_png.exists()


def test_stage_training_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    random_state_factory: Callable[[int | None], np.random.RandomState],
) -> None:
    from src.training.main import main as train_main

    rng = random_state_factory(3)
    base = _mock_df(30, rng)
    base.columns = ["weight-(kg)", "physical-exercise", "gender", "age"]
    train_df = base.sample(20, random_state=3).reset_index(drop=True)
    test_df = base.drop(index=list(train_df.index)).reset_index(drop=True)

    called = {"train": False, "save": False, "summary": False}
    best_params = {"n_estimators": 64, "learning_rate": 0.1}
    out_model_holder: dict[str, str | None] = {"p": None}

    def _fake_train(
        parquet_path: str, params_json_path: str, *, target_column: str, random_state: int
    ):
        called["train"] = True

        class _M:
            def predict(self, X):
                return np.zeros(len(X))

        return _M()

    def _fake_save(model: Any, model_path: str) -> str:
        called["save"] = True
        fd, p = tempfile.mkstemp(suffix=".joblib", dir=str(tmp_path))
        os.close(fd)
        Path(p).write_bytes(b"JOBLIB")
        out_model_holder["p"] = p
        return p

    def _fake_summary(payload: dict, json_path: Path) -> Path:
        called["summary"] = True
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        return json_path

    monkeypatch.setattr("src.training.main.train_lightgbm_with_best_params", _fake_train)
    monkeypatch.setattr("src.training.main.save_lightgbm_model", _fake_save)
    monkeypatch.setattr(
        "src.training.main.load_splits_from_parquet", lambda _p: (train_df, test_df)
    )
    monkeypatch.setattr("src.training.main.read_best_params", lambda _p: best_params)
    monkeypatch.setattr("src.training.main.save_training_results", _fake_summary)

    rc = train_main(
        [
            "--parquet",
            str(tmp_path / "data" / "dataset_splits_encoded.parquet"),
            "--params",
            str(tmp_path / "results" / "best_lightgbm_params.json"),
            "--out",
            str((tmp_path / "results" / "models" / "lightgbm_temp").with_suffix("")),
            "--target-column",
            "weight-(kg)",
            "--random-state",
            "123",
            "--no-random-forest",
        ]
    )
    assert rc == 0 and all(called.values()) and out_model_holder["p"] is not None


def test_stage_eval_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    random_state_factory: Callable[[int | None], np.random.RandomState],
) -> None:
    from src.eval.main import main as eval_main

    rng = random_state_factory(4)
    base = _mock_df(30, rng)
    base.columns = ["weight-(kg)", "physical-exercise", "gender", "age"]
    train_df = base.sample(20, random_state=4).reset_index(drop=True)
    test_df = base.drop(index=list(train_df.index)).reset_index(drop=True)

    eval_metrics = {
        "mae": 0.1,
        "mse": 0.01,
        "rmse": 0.1,
        "r2": 0.9,
        "mape": 0.05,
        "median_ae": 0.08,
        "explained_variance": 0.88,
    }

    def _eval_stub(
        *,
        parquet_path: Path,  # noqa: ARG001
        model_path: Path,  # noqa: ARG001
        target_column: str,  # noqa: ARG001
        shap_output_dir: Path,
        shap_max_display: int,
        model_label: str,
        shap_sample_size: int | None,
        shap_random_state: int,
    ) -> dict[str, Any]:
        assert isinstance(shap_output_dir, Path)
        assert shap_max_display >= 0
        assert isinstance(shap_random_state, int)
        if model_label == "random_forest":
            return {
                "mae": 0.2,
                "mse": 0.04,
                "rmse": 0.2,
                "r2": 0.8,
                "mape": 0.1,
                "median_ae": 0.15,
                "explained_variance": 0.75,
                "shap": None,
            }
        return {**eval_metrics, "shap": None}

    monkeypatch.setattr("src.eval.main.evaluate_lightgbm_on_test", _eval_stub)
    monkeypatch.setattr("src.eval.main.load_splits_from_parquet", lambda _p: (train_df, test_df))
    monkeypatch.setattr("src.eval.main.ensure_numeric_columns", lambda X: X)
    monkeypatch.setitem(
        __import__("sys").modules,
        "joblib",
        __import__("types").SimpleNamespace(load=lambda _p: object()),
    )

    eval_dir = tmp_path / "results" / "eval"
    expected_json_path = eval_dir / "lightgbm_test_metrics.json"

    def _save_results(payload: dict, json_path: Path) -> Path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        return json_path

    monkeypatch.setattr("src.eval.main.save_training_results", _save_results)

    capsys.readouterr()
    caplog.set_level(logging.INFO)
    caplog.clear()

    out_model = tmp_path / "results" / "models" / "lightgbm.joblib"
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_model.write_bytes(b"JOBLIB")

    rc = eval_main(
        [
            "--parquet",
            str(tmp_path / "data" / "dataset_splits_encoded.parquet"),
            "--model",
            str(out_model),
            "--target-column",
            "weight-(kg)",
            "--batch-size",
            "2",
            "--n-jobs",
            "-1",
            "--eval-dir",
            str(eval_dir),
        ]
    )
    assert rc == 0

    captured = capsys.readouterr()
    messages = []
    for line in captured.out.strip().split("\n"):
        parts = line.split(" | ", 3)
        if len(parts) == 4 and parts[2] == "src.eval.main":
            messages.append(parts[3])

    metrics_message = next((m for m in messages if m.startswith("LightGBM metrics: ")), None)
    assert metrics_message is not None
    assert json.loads(metrics_message.split("LightGBM metrics: ", 1)[1]) == eval_metrics
    assert any(
        m.startswith("LightGBM evaluation artifact confirmed at ") and str(expected_json_path) in m
        for m in messages
    )


def test_stage_baseline_ridge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.baseline.main import main as ridge_main

    params_json = tmp_path / "results" / "best_ridge_params.json"
    ridge_model_out = tmp_path / "results" / "models" / "ridge"
    ridge_scaler_out = tmp_path / "results" / "models" / "ridge_scaler.joblib"

    # Patch the symbols imported in baseline.main (direct imports)
    monkeypatch.setattr(
        "src.baseline.main.optimize_ridge_hyperparameters",
        lambda *a, **k: ({"alpha": 1.0}, 1.23, {"cv_mse_mean": 1.23}),
    )

    class _M:
        def predict(self, X):
            return np.zeros(len(X))

    class _S:
        def transform(self, X):
            return X

    monkeypatch.setattr(
        "src.baseline.main.train_ridge_with_best_params",
        lambda *a, **k: (_M(), _S()),
    )

    rc_opt = ridge_main(
        [
            "--parquet",
            str(tmp_path / "data" / "dataset_splits_encoded.parquet"),
            "--target-column",
            "weight-(kg)",
            "--optimize",
            "--params-out",
            str(params_json),
        ]
    )
    assert rc_opt == 0 and params_json.exists()

    rc_train = ridge_main(
        [
            "--parquet",
            str(tmp_path / "data" / "dataset_splits_encoded.parquet"),
            "--target-column",
            "weight-(kg)",
            "--train",
            "--params",
            str(params_json),
            "--model-out",
            str(ridge_model_out),
            "--scaler-out",
            str(ridge_scaler_out),
        ]
    )
    assert rc_train == 0
    assert Path(str(ridge_model_out).rstrip(".joblib") + ".joblib").exists()
    assert ridge_scaler_out.exists()


if __name__ == "__main__":  # pragma: no cover - convenience
    pytest.main([__file__, "-v"])
