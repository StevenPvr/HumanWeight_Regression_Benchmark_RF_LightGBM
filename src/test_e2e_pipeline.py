from __future__ import annotations

"""End-to-end (E2E) pipeline test using mocked data and lightweight stubs.

WHY: Validate the orchestration across all pipeline entrypoints (data_cleaning,
data_preparation, hyperparameters_optimization, feature_engineering, training,
evaluation) without relying on real data or expensive computations. This
keeps the test fast, deterministic, and self-contained.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

try:
    from src.constants import TARGET_COLUMN
except ModuleNotFoundError:  # pragma: no cover
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.constants import TARGET_COLUMN


def test_e2e_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Run the full pipeline end-to-end with mocks and assert artifacts exist."""

    # ----------------------------
    # 0) Synthetic data source
    # ----------------------------
    rng = np.random.RandomState(123)
    raw_df = pd.DataFrame(
        {
            "Weight (kg)": rng.normal(loc=70, scale=10, size=48),
            "physical exercise": rng.randint(0, 2, size=48),
            "gender": rng.choice(["M", "F"], size=48),
            "age": rng.randint(18, 70, size=48),
        }
    )

    # ----------------------------
    # 1) data_cleaning.main
    # ----------------------------
    from src.data_cleaning.main import main as cleaning_main

    # Replace CSV source with synthetic df
    monkeypatch.setattr(
        "src.data_cleaning.main.pd.read_csv", lambda _path: raw_df.copy()
    )

    # Redirect saving to tmp outputs
    cleaned_dir = tmp_path / "data"
    cleaned_csv = cleaned_dir / "dataset_cleaned_final.csv"
    cleaned_parquet = cleaned_dir / "dataset_cleaned_final.parquet"

    def _fake_save_cleaned(
        dataset: pd.DataFrame,
        output_name: str = "dataset_cleaned_final",
        output_dir: Path | str | None = None,
    ) -> None:
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(cleaned_csv, index=False)
        dataset.to_parquet(cleaned_parquet, index=False)

    monkeypatch.setattr("src.data_cleaning.main.save_cleaned_dataset", _fake_save_cleaned)

    # Stub matplotlib usage to avoid real Axes dependence
    class _Fig:
        def savefig(self, path, dpi=150):
            # WHY: Redirect plot outputs into tmp_path to keep repo clean.
            safe_dir = tmp_path / "plots"
            safe_dir.mkdir(parents=True, exist_ok=True)
            (safe_dir / Path(str(path)).name).write_bytes(b"PNG")

        def tight_layout(self):
            return None

        def add_subplot(self, *_args, **_kwargs):  # pragma: no cover - safety
            return _Axes(self)

        def get_axes(self):  # pragma: no cover - safety
            return []

    class _Axis:
        def set_visible(self, *_args, **_kwargs):  # pragma: no cover - safety
            return None

    class _Axes:
        def __init__(self, fig: _Fig):
            self._fig = fig

        def get_figure(self):  # pragma: no cover - safety
            return self._fig

        def get_yaxis(self):  # pragma: no cover - safety
            return _Axis()

        def get_xaxis(self):  # pragma: no cover - safety
            return _Axis()

        # Methods used by data_cleaning.main after plotting
        def set_title(self, *_args, **_kwargs):  # pragma: no cover - safety
            return None

        def set_xlabel(self, *_args, **_kwargs):  # pragma: no cover - safety
            return None

        def set_ylabel(self, *_args, **_kwargs):  # pragma: no cover - safety
            return None

        def set_xticklabels(self, *_args, **_kwargs):  # pragma: no cover - safety
            return None

    class _PLT:
        def subplots(self, *args, **kwargs):
            fig = _Fig()
            return fig, _Axes(fig)

        def close(self, *args, **_kwargs):
            return None

    monkeypatch.setattr("src.data_cleaning.main.plt", _PLT())

    # Short-circuit pandas -> matplotlib plotting path to avoid full Matplotlib
    import pandas.plotting._core as _pd_plot_core  # type: ignore

    monkeypatch.setattr(
        _pd_plot_core.PlotAccessor, "__call__", lambda self, *args, **kwargs: None
    )

    cleaning_main()
    assert cleaned_csv.exists() and cleaned_parquet.exists()

    # ----------------------------
    # 2) data_preparation.main
    # ----------------------------
    from src.data_preparation.main import main as prep_main

    # Provide normalized columns as if produced by cleaning
    cleaned_df = raw_df.copy()
    cleaned_df.columns = [TARGET_COLUMN, "physical-exercise", "gender", "age"]

    monkeypatch.setattr(
        "src.data_preparation.main.pd.read_csv", lambda _p: cleaned_df.copy()
    )

    splits_dir = tmp_path / "data"
    splits_csv = splits_dir / "dataset_splits_encoded.csv"
    splits_parquet = splits_dir / "dataset_splits_encoded.parquet"

    def _fake_save_splits(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        csv_path: str,
        parquet_path: str,
    ) -> None:
        # WHY: Persist to tmp while keeping the contract expected by downstream steps
        splits_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(
            [
                train_df.assign(split="train"),
                val_df.assign(split="val"),
                test_df.assign(split="test"),
            ]
        ).to_csv(splits_csv, index=False)
        pd.concat(
            [
                train_df.assign(split="train"),
                val_df.assign(split="val"),
                test_df.assign(split="test"),
            ]
        ).to_parquet(splits_parquet, index=False)

    monkeypatch.setattr("src.data_preparation.main.save_splits_with_marker", _fake_save_splits)
    monkeypatch.setattr("src.data_preparation.main.save_label_encoders_mappings", lambda enc, j, c: None)

    prep_main()
    assert splits_csv.exists() and splits_parquet.exists()

    # Construct small splits for downstream mocks
    train_df = cleaned_df.sample(n=28, random_state=123).reset_index(drop=True)
    # WHY: Convert pandas Index to list to satisfy static type checkers for DataFrame.drop
    val_df = cleaned_df.drop(index=list(train_df.index)).sample(n=10, random_state=123).reset_index(drop=True)
    test_df = cleaned_df.drop(index=list(train_df.index.union(val_df.index))).reset_index(drop=True)

    # ----------------------------
    # 3) hyperparameters_optimization.main
    # ----------------------------
    from src.hyperparameters_optimization.main import main as hpo_main
    # Redirect hyperopt results directory to a temp location
    monkeypatch.setattr("src.hyperparameters_optimization.main.RESULTS_DIR", tmp_path / "results")

    monkeypatch.setattr(
        "src.hyperparameters_optimization.main.load_splits_from_parquet",
        lambda _p: (train_df, val_df, test_df),
    )

    best_params = {"n_estimators": 64, "learning_rate": 0.1}
    val_summary = {"val_mse": 1.0}

    # Save a minimal best params JSON to tmp so training can consume it
    params_dir = tmp_path / "results"
    params_json = params_dir / "best_lightgbm_params.json"

    def _fake_optimize(parquet_path: str, target_column: str, n_trials: int, random_state: int):
        params_dir.mkdir(parents=True, exist_ok=True)
        with params_json.open("w", encoding="utf-8") as handle:
            json.dump({"best_params": best_params}, handle)
        return best_params, 1.0, val_summary

    monkeypatch.setattr(
        "src.hyperparameters_optimization.main.optimize_lightgbm_hyperparameters",
        _fake_optimize,
    )
    # Avoid running real RF hyperopt; return a small stub quickly
    monkeypatch.setattr(
        "src.hyperparameters_optimization.main.optimize_random_forest_hyperparameters",
        lambda *args, **kwargs: ({"n_estimators": 10, "max_depth": 3}, 1.23, {"val_mse": 1.23}),
    )
    
    # WHY: Prevent hpo_main from inheriting pytest's argv; provide minimal arguments
    monkeypatch.setattr(
        "src.hyperparameters_optimization.main.sys.argv",
        ["pytest", "--models", "both"],
    )

    hpo_main()  # Use defaults; our stubs control inputs/outputs
    assert params_json.exists()

    # ----------------------------
    # 4) feature_engineering.main
    # ----------------------------
    from src.feature_engineering.main import main as fe_main

    monkeypatch.setattr(
        "src.feature_engineering.main.load_splits_from_parquet",
        lambda _p: (train_df, val_df, test_df),
    )

    # Minimal permutation importance DataFrame
    importance_df = pd.DataFrame(
        {
            "feature": ["age", "gender"],
            "importance_mean": [0.1, 0.05],
            "importance_std": [0.01, 0.005],
        }
    )
    out_json = tmp_path / "results" / "feature_importances.json"
    out_png = tmp_path / "plots" / "permutation" / "feature_importances.png"

    monkeypatch.setattr(
        "src.feature_engineering.main.compute_random_forest_permutation_importance",
        lambda *args, **kwargs: importance_df,
    )
    monkeypatch.setattr(
        "src.feature_engineering.main.save_feature_importances_to_json",
        lambda df, path: (
            os.makedirs(os.path.dirname(str(out_json)), exist_ok=True),
            Path(str(out_json)).write_text(
                json.dumps(df.to_dict(orient="records")), encoding="utf-8"
            ),
        ),
    )
    monkeypatch.setattr(
        "src.feature_engineering.main.plot_feature_importances",
        lambda df, path, top_n=None: (
            os.makedirs(os.path.dirname(str(out_png)), exist_ok=True),
            Path(str(out_png)).write_bytes(b"PNG"),
        ),
    )

    # WHY: Prevent fe_main from inheriting pytest's argv; provide required CLI args
    monkeypatch.setattr(
        "src.feature_engineering.main.sys.argv",
        [
            "pytest",
            "--input",
            str(splits_parquet),
            "--target",
            TARGET_COLUMN,
            "--output-json",
            str(out_json),
            "--output-plot",
            str(out_png),
        ],
    )

    fe_main()
    assert out_json.exists() and out_png.exists()

    # ----------------------------
    # 5) training.main (CLI-style call)
    # ----------------------------
    from src.training.main import main as train_main

    out_model = tmp_path / "results" / "models" / "lightgbm.joblib"
    called = {"train": False, "save": False}

    def _fake_train(
        parquet_path: str, params_json_path: str, *, target_column: str, random_state: int
    ):
        # WHY: Avoid real training; return a dummy model with predict API
        called["train"] = True

        class _M:
            def predict(self, X):  # type: ignore[no-untyped-def]
                return np.zeros(len(X))

        return _M()

    def _fake_save(model: Any, model_path: str) -> str:
        called["save"] = True
        os.makedirs(os.path.dirname(str(out_model)), exist_ok=True)
        Path(str(out_model)).write_bytes(b"JOBLIB")
        return str(out_model)

    monkeypatch.setattr(
        "src.training.main.train_lightgbm_with_best_params", _fake_train
    )
    monkeypatch.setattr("src.training.main.save_lightgbm_model", _fake_save)

    # Keep logs out of captured stdout parsing downstream
    capsys.readouterr()

    rc = train_main(
        [
            "--parquet",
            str(splits_parquet),
            "--params",
            str(params_json),
            "--out",
            str(out_model.with_suffix("")),
            "--target-column",
            TARGET_COLUMN,
            "--random-state",
            "123",
            "--no-random-forest",  # ensure no persistent RF artifacts are created
        ]
    )
    assert rc == 0 and out_model.exists() and all(called.values())

    # ----------------------------
    # 6) eval.main (CLI-style call)
    # ----------------------------
    from src.eval.main import main as eval_main

    eval_dir = tmp_path / "results" / "eval"
    expected_json_path = eval_dir / "lightgbm_test_metrics.json"

    # Lightweight stubs for evaluation
    eval_metrics = {
        "mae": 0.1,
        "mse": 0.01,
        "rmse": 0.1,
        "r2": 0.9,
        "mape": 0.05,
        "median_ae": 0.08,
        "explained_variance": 0.88,
    }

    def _fake_evaluate_lightgbm_on_test(
        *,
        parquet_path: str,
        model_path: str,
        target_column: str,
        shap_output_dir: Path,
        shap_max_display: int,
        model_label: str,
        shap_sample_size: int | None,
        shap_random_state: int,
    ) -> dict[str, Any]:
        """Accept CLI's extended evaluation signature and yield canned metrics."""

        # WHY: Honor new SHAP-related parameters while avoiding heavy dependencies.
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
                "shap": None,
            }
        assert shap_sample_size is None
        assert model_label == "lightgbm"
        return {**eval_metrics, "shap": None}

    monkeypatch.setattr(
        "src.eval.main.evaluate_lightgbm_on_test",
        _fake_evaluate_lightgbm_on_test,
    )
    monkeypatch.setattr(
        "src.eval.main.load_splits_from_parquet", lambda _p: (train_df, val_df, test_df)
    )
    monkeypatch.setattr("src.eval.main.ensure_numeric_columns", lambda X: X)

    monkeypatch.setattr(
        "src.eval.main.save_training_results",
        lambda payload, json_path: (
            json_path.parent.mkdir(parents=True, exist_ok=True),
            json_path.write_text(json.dumps(payload), encoding="utf-8"),
            json_path,
        )[-1],
    )

    # Clear previous outputs; capture only eval prints
    capsys.readouterr()
    rc = eval_main(
        [
            "--parquet",
            str(splits_parquet),
            "--model",
            str(out_model),
            "--target-column",
            TARGET_COLUMN,
            "--batch-size",
            "2",
            "--n-jobs",
            "-1",
            "--output-dir",
            str(eval_dir),
        ]
    )
    assert rc == 0 and expected_json_path.exists()

    # WHY: When running standalone, logs go to stdout; when running with pytest, caplog captures them
    # Try caplog first (multi-test scenario), fall back to stdout parsing (standalone scenario)
    messages = []
    
    # Check caplog first
    caplog_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "src.eval.main"
    ]
    
    if caplog_messages:
        messages = caplog_messages
    else:
        # Fall back to parsing stdout
        captured = capsys.readouterr()
        output_lines = captured.out.strip().split("\n")
        
        for line in output_lines:
            # Format: "timestamp | level | logger_name | message"
            parts = line.split(" | ", 3)
            if len(parts) == 4 and parts[2] == "src.eval.main":
                messages.append(parts[3])
    
    # Find and validate LightGBM metrics message
    metrics_message = next((msg for msg in messages if msg.startswith("LightGBM metrics: ")), None)
    assert metrics_message is not None, f"Expected 'LightGBM metrics:' message not found in: {messages}"
    
    # Extract JSON from the log message
    first_json = json.loads(metrics_message.split("LightGBM metrics: ", 1)[1])
    assert first_json == eval_metrics
    
    # Validate that the summary path is mentioned in logs
    summary_message = next((msg for msg in messages if msg.startswith("LightGBM evaluation artifact confirmed at ")), None)
    assert summary_message is not None and str(expected_json_path) in summary_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
