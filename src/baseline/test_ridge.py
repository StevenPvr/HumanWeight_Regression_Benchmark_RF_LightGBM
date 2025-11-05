from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
from pathlib import Path


def test_cli_optimize_and_train(monkeypatch, tmp_path: Path) -> None:
    # Synthetic parquet-like splits paths
    parquet_path = tmp_path / "data" / "splits.parquet"
    params_out = tmp_path / "results" / "best_ridge_params.json"
    model_out = tmp_path / "results" / "models" / "ridge"
    scaler_out = tmp_path / "results" / "models" / "ridge_scaler.joblib"

    # Minimal data to satisfy code paths (not actually read here)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.write_bytes(b"PARQUET_MOCK")

    # Stubs for optimization and training
    called: dict[str, bool] = {"opt": False, "train": False}

    def _fake_optimize(*args, **kwargs):
        called["opt"] = True
        return {"alpha": 1.0}, 1.23, {"cv_mse": 1.23}

    class _M:
        def predict(self, X):
            n = len(X) if hasattr(X, "__len__") else 1
            return [0.0] * int(n)

    class _S:
        mean_ = [0.0]
        scale_ = [1.0]

    def _fake_train(*args, **kwargs):
        called["train"] = True
        return _M(), _S()

    monkeypatch.setattr("src.baseline.ridge.optimize_ridge_hyperparameters", _fake_optimize)
    monkeypatch.setattr("src.baseline.ridge.train_ridge_with_best_params", _fake_train)

    # Save results writes JSON; bypass real save with a simple writer using the same signature
    from src.baseline import main as ridge_main

    # Run optimize
    rc = ridge_main.main(
        [
            "--parquet",
            str(parquet_path),
            "--target-column",
            "weight-(kg)",
            "--optimize",
            "--n-trials",
            "2",
            "--params-out",
            str(params_out),
        ]
    )
    assert rc == 0 and called["opt"] and params_out.exists()
    payload = json.loads(params_out.read_text(encoding="utf-8"))
    assert "best_params" in payload and payload["best_params"]["alpha"] == 1.0

    # Run train
    rc = ridge_main.main(
        [
            "--parquet",
            str(parquet_path),
            "--target-column",
            "weight-(kg)",
            "--train",
            "--params",
            str(params_out),
            "--model-out",
            str(model_out),
            "--scaler-out",
            str(scaler_out),
        ]
    )
    assert rc == 0 and called["train"]


if __name__ == "__main__":
    import pytest  # pragma: no cover

    pytest.main([__file__, "-q"])  # pragma: no cover
