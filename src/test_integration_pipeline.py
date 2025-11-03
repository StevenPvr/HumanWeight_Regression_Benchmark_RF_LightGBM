from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest


def _make_mock_df(n: int, rng: np.random.RandomState) -> pd.DataFrame:
    # WHY: Minimal synthetic dataset to flow through pipeline without heavy compute
    df = pd.DataFrame({
        "Weight (kg)": rng.normal(loc=70, scale=10, size=n),
        "physical exercise": rng.randint(0, 2, size=n),
        "gender": rng.choice(["M", "F"], size=n),
        "age": rng.randint(18, 70, size=n),
    })
    return df


def test_integration_pipeline_main_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Step order: data_cleaning -> data_preparation -> hyperparameters_optimization -> feature_engineering -> training -> eval
    # NOTE: There is no 'models.main' in the repo; skipping that step.

    # 0) Synthetic data
    rng = np.random.RandomState(42)
    raw_df = _make_mock_df(40, rng)

    # ----------------------------
    # 1) data_cleaning.main
    # ----------------------------
    from src.data_cleaning.main import main as cleaning_main

    # Read CSV replaced with synthetic df
    monkeypatch.setattr("src.data_cleaning.main.pd.read_csv", lambda path: raw_df.copy())

    # Save cleaned dataset redirected to tmp and validated
    cleaned_out = {
        "called": False,
        "csv": tmp_path / "data" / "dataset_cleaned_final.csv",
        "parquet": tmp_path / "data" / "dataset_cleaned_final.parquet",
    }

    def _fake_save_cleaned(dataset: pd.DataFrame, output_name: str = "dataset_cleaned_final", output_dir: Path | str | None = None) -> None:
        cleaned_out["called"] = True
        out_dir = tmp_path / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(cleaned_out["csv"], index=False)
        dataset.to_parquet(cleaned_out["parquet"], index=False)

    monkeypatch.setattr("src.data_cleaning.main.save_cleaned_dataset", _fake_save_cleaned)

    # Stub matplotlib usage
    class _Fig:
        def savefig(self, path, dpi=150):
            # WHY: Redirect plot writes to tmp_path to avoid polluting the repo.
            safe_dir = tmp_path / "plots"
            safe_dir.mkdir(parents=True, exist_ok=True)
            (safe_dir / Path(str(path)).name).write_bytes(b"PNG")
        def tight_layout(self):
            return None
        def add_subplot(self, *_args, **_kwargs):
            # Return a dummy axes if pandas ever requests it
            return _Axes(self)
        def get_axes(self):
            # Return empty to satisfy pandas adornment without real Matplotlib
            return []

    class _Axis:
        def set_visible(self, *_args, **_kwargs):
            return None

    class _Axes:
        def __init__(self, fig: _Fig):
            self._fig = fig
        def get_figure(self):
            return self._fig
        def get_yaxis(self):
            return _Axis()
        def get_xaxis(self):
            return _Axis()
        def get_legend(self):
            # Signal no existing legend so pandas can create one if needed
            return None
        def legend(self, *_args, **_kwargs):
            # Return a minimal object that pandas won't interact with further
            class _Legend:
                def remove(self):
                    return None
            return _Legend()
        def grid(self, *_args, **_kwargs):
            return None
        def set_xticks(self, *_args, **_kwargs):
            return None
        def set_yticks(self, *_args, **_kwargs):
            return None
        def set_xlim(self, *_args, **_kwargs):
            return None
        def set_ylim(self, *_args, **_kwargs):
            return None
        def bar(self, *_args, **_kwargs):
            # Minimal stub for pandas' bar plotting
            return None
        def set_title(self, *_args, **_kwargs):
            return None
        def set_xlabel(self, *_args, **_kwargs):
            return None
        def set_ylabel(self, *_args, **_kwargs):
            return None
        def set_xticklabels(self, *_args, **_kwargs):
            return None

    class _PLT:
        def subplots(self, *args, **kwargs):
            fig = _Fig()
            return fig, _Axes(fig)
        def close(self, *args, **_kwargs):
            return None

    monkeypatch.setattr("src.data_cleaning.main.plt", _PLT())

    # Short-circuit pandas -> matplotlib plotting path to avoid requiring real Axes API
    # WHY: Pandas' PlotAccessor expects full Matplotlib Axes (xaxis/yaxis, etc.).
    # The integration test validates pipeline orchestration, not visualization internals.
    import pandas.plotting._core as _pd_plot_core  # type: ignore
    monkeypatch.setattr(_pd_plot_core.PlotAccessor, "__call__", lambda self, *args, **kwargs: None)

    cleaning_main()
    assert cleaned_out["called"]

    # ----------------------------
    # 2) data_preparation.main
    # ----------------------------
    from src.data_preparation.main import main as prep_main

    # Provide normalized columns as if from cleaning
    cleaned_df = raw_df.copy()
    cleaned_df.columns = [
        "weight-(kg)",
        "physical-exercise",
        "gender",
        "age",
    ]
    monkeypatch.setattr("src.data_preparation.main.pd.read_csv", lambda path: cleaned_df.copy())

    # Redirect split/encoder saving to tmp
    splits_out = {"csv": tmp_path / "data" / "dataset_splits_encoded.csv", "parquet": tmp_path / "data" / "dataset_splits_encoded.parquet"}

    def _fake_save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, csv_path: str, parquet_path: str) -> None:
        # Write minimal outputs to tmp to keep pipeline happy if needed downstream
        Path(splits_out["csv"]).parent.mkdir(parents=True, exist_ok=True)
        pd.concat([
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ]).to_csv(splits_out["csv"], index=False)
        pd.concat([
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ]).to_parquet(splits_out["parquet"], index=False)

    monkeypatch.setattr("src.data_preparation.main.save_splits_with_marker", _fake_save_splits)
    monkeypatch.setattr("src.data_preparation.main.save_label_encoders_mappings", lambda enc, j, c: None)

    prep_main()

    # Build small splits for later steps
    # Keep target numeric and include one categorical encoded column
    train_df = cleaned_df.sample(n=24, random_state=123).reset_index(drop=True)
    # WHY: Convert pd.Index to a plain list for static type checker compatibility (Pyright expects IndexLabel)
    val_df = cleaned_df.drop(index=list(train_df.index)).sample(n=8, random_state=123).reset_index(drop=True)
    test_df = cleaned_df.drop(index=list(train_df.index.union(val_df.index))).reset_index(drop=True)

    # ----------------------------
    # 3) hyperparameters_optimization.main
    # ----------------------------
    from src.hyperparameters_optimization.main import main as hpo_main
    # Redirect hyperopt outputs to tmp results directory to avoid permanent writes
    monkeypatch.setattr("src.hyperparameters_optimization.main.RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr("src.hyperparameters_optimization.main.load_splits_from_parquet", lambda p: (train_df, val_df, test_df))
    best_params = {"n_estimators": 64, "learning_rate": 0.1}
    val_summary = {"val_mse": 1.0}
    monkeypatch.setattr(
        "src.hyperparameters_optimization.main.optimize_lightgbm_hyperparameters",
        lambda parquet_path, target_column, n_trials, random_state: (
            best_params,
            1.0,
            val_summary,
        ),
    )
    # New: default pipeline now runs RF optimization after LightGBM; stub it out
    monkeypatch.setattr(
        "src.hyperparameters_optimization.main.optimize_random_forest_hyperparameters",
        lambda parquet_path, target_column, n_trials, random_state: (
            {"n_estimators": 50, "max_depth": 5}, 1.1, {"val_mse": 1.1}
        ),
    )
    monkeypatch.setattr(
        "src.hyperparameters_optimization.main.sys.argv",
        ["pytest", "--models", "both"],
    )
    monkeypatch.setattr(
        "src.hyperparameters_optimization.main.sys.stdin",
        SimpleNamespace(isatty=lambda: False),
    )
    hpo_main()

    # ----------------------------
    # 4) feature_engineering.main
    # ----------------------------
    from src.feature_engineering.main import main as fe_main
    monkeypatch.setattr("src.feature_engineering.main.load_splits_from_parquet", lambda p: (train_df, val_df, test_df))

    # Minimal permutation importance DataFrame
    importance_df = pd.DataFrame({
        "feature": ["age", "gender"],
        "importance_mean": [0.1, 0.05],
        "importance_std": [0.01, 0.005],
    })
    monkeypatch.setattr("src.feature_engineering.main.compute_random_forest_permutation_importance", lambda *args, **kwargs: importance_df)
    out_json = tmp_path / "results" / "feature_importances.json"
    out_png = tmp_path / "plots" / "permutation" / "feature_importances.png"
    monkeypatch.setattr("src.feature_engineering.main.save_feature_importances_to_json", lambda df, path: (os.makedirs(os.path.dirname(str(out_json)), exist_ok=True), Path(str(out_json)).write_text(json.dumps(df.to_dict(orient='records')), encoding='utf-8')))
    monkeypatch.setattr("src.feature_engineering.main.plot_feature_importances", lambda df, path, top_n=None: (os.makedirs(os.path.dirname(str(out_png)), exist_ok=True), Path(str(out_png)).write_bytes(b"PNG")))
    monkeypatch.setattr(
        "src.feature_engineering.main.sys.argv",
        [
            "pytest",
            "--input",
            str(splits_out["parquet"]),
            "--target",
            "weight-(kg)",
            "--output-json",
            str(out_json),
            "--output-plot",
            str(out_png),
        ],
    )
    fe_main()

    # ----------------------------
    # 5) training.main
    # ----------------------------
    from src.training.main import main as train_main
    called = {"train": False, "save": False, "summary": False}
    model_artifact: dict[str, str | None] = {"path": None}

    def _fake_train(parquet_path: str, params_json_path: str, *, target_column: str, random_state: int):
        called["train"] = True
        class _M:
            def predict(self, X):
                return np.zeros(len(X))
        return _M()

    def _fake_save(model: Any, model_path: str) -> str:
        called["save"] = True
        # WHY: Persist into a dedicated temporary file to avoid colliding with real artifacts.
        handle, temp_path = tempfile.mkstemp(suffix=".joblib", dir=str(tmp_path))
        os.close(handle)
        Path(temp_path).write_bytes(b"JOBLIB")
        model_artifact["path"] = temp_path
        return temp_path

    def _fake_save_summary(payload: dict, json_path: Path) -> Path:
        called["summary"] = True
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        return json_path

    monkeypatch.setattr("src.training.main.train_lightgbm_with_best_params", _fake_train)
    monkeypatch.setattr("src.training.main.save_lightgbm_model", _fake_save)
    monkeypatch.setattr("src.training.main.load_splits_from_parquet", lambda p: (train_df, val_df, test_df))
    monkeypatch.setattr("src.training.main.read_best_params", lambda p: best_params)
    monkeypatch.setattr("src.training.main.save_training_results", _fake_save_summary)

    rc = train_main([
        "--parquet", str(splits_out["parquet"]),
        "--params", str(tmp_path / "results" / "best_lightgbm_params.json"),
        "--out", str((tmp_path / "results" / "models" / "lightgbm_temp").with_suffix("")),
        "--target-column", "weight-(kg)",
        "--random-state", "123",
        "--no-random-forest",  # avoid unintended persistent RF artifacts
    ])
    assert rc == 0 and all(called.values())
    assert model_artifact["path"] is not None
    out_model = Path(model_artifact["path"])

    # ----------------------------
    # 6) eval.main
    # ----------------------------
    from src.eval.main import main as eval_main
    # Lightweight stubs as in dedicated eval tests
    eval_metrics = {"mae": 0.1, "mse": 0.01, "rmse": 0.1, "r2": 0.9, "mape": 0.05, "median_ae": 0.08, "explained_variance": 0.88}

    def _fake_evaluate_lightgbm_on_test(
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
        """Accept extended evaluation signature while returning canned metrics."""

        # WHY: Ensure the monkeypatched stub validates new SHAP-related parameters without heavy work.
        assert isinstance(shap_output_dir, Path)
        assert shap_max_display >= 0
        assert isinstance(shap_random_state, int)
        if model_label == "random_forest":
            assert shap_max_display == 0
            assert shap_sample_size is None
            return {
                "mae": 0.2,
                "mse": 0.04,
                "rmse": 0.2,
                "r2": 0.8,
                "mape": 0.1,
                "median_ae": 0.15,
                "explained_variance": 0.75,
                "y_std": 1.1,
                "pred_std": 0.9,
                "residual_std": 0.35,
                "shap": None,
            }
        assert shap_sample_size is None
        assert model_label == "lightgbm"
        return {**eval_metrics, "shap": None}

    monkeypatch.setattr("src.eval.main.evaluate_lightgbm_on_test", _fake_evaluate_lightgbm_on_test)
    monkeypatch.setattr("src.eval.main.load_splits_from_parquet", lambda p: (train_df, val_df, test_df))
    monkeypatch.setattr("src.eval.main.ensure_numeric_columns", lambda X: X)
    monkeypatch.setitem(__import__("sys").modules, "joblib", __import__("types").SimpleNamespace(load=lambda p: object()))

    eval_dir = tmp_path / "results" / "eval"
    expected_json_path = eval_dir / "lightgbm_test_metrics.json"
    monkeypatch.setattr(
        "src.eval.main.save_training_results",
        lambda payload, json_path: (
            json_path.parent.mkdir(parents=True, exist_ok=True),
            json_path.write_text(json.dumps(payload), encoding="utf-8"),
            json_path,
        )[-1],
    )

    # Clear previous captured stdout so we only parse eval outputs below
    capsys.readouterr()
    caplog.set_level(logging.INFO)
    caplog.clear()

    rc = eval_main([
        "--parquet", str(splits_out["parquet"]),
        "--model", str(out_model),
        "--target-column", "weight-(kg)",
        "--batch-size", "2",
        "--n-jobs", "-1",
        "--eval-dir", str(eval_dir),
    ])
    assert rc == 0
    
    # WHY: caplog may not capture records when custom handlers write to stdout.
    # Parse captured stdout output instead for robust log message validation.
    captured = capsys.readouterr()
    output_lines = captured.out.strip().split("\n")
    
    # Extract log messages from formatted output
    messages = []
    for line in output_lines:
        # Format: "timestamp | level | logger_name | message"
        parts = line.split(" | ", 3)
        if len(parts) == 4 and parts[2] == "src.eval.main":
            messages.append(parts[3])
    
    metrics_message = next((msg for msg in messages if msg.startswith("LightGBM metrics: ")), None)
    assert metrics_message is not None, f"Expected 'LightGBM metrics:' message not found in: {messages}"
    logged_metrics = json.loads(metrics_message.split("LightGBM metrics: ", 1)[1])
    assert logged_metrics == eval_metrics
    summary_message = next((msg for msg in messages if msg.startswith("LightGBM summary saved to ")), None)
    assert summary_message is not None and str(expected_json_path) in summary_message
    artifact_message = next((msg for msg in messages if msg.startswith("LightGBM evaluation artifact confirmed at ")), None)
    assert artifact_message is not None and str(expected_json_path) in artifact_message

    if out_model.exists():
        out_model.unlink()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
