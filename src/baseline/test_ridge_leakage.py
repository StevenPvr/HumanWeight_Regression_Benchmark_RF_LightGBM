from __future__ import annotations

import json
from typing import Any, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_optuna_cv_encodes_per_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure encoding is fitted per-fold (no global fit leakage).

    Strategy:
    - Monkeypatch KFold to yield controlled splits (3 folds).
    - Monkeypatch OneHotEncoder to record categories seen at each fit call.
    - Assert there is exactly one fit per fold and that the categories passed
      to fit equal the categories present in that fold's train subset.
    """
    from src.baseline import ridge as ridge_mod

    # Build a synthetic train-only dataset (test split unused during CV)
    n_folds = 3
    idx = np.arange(12)
    # Pre-assign folds deterministically: 0..3, 4..7, 8..11
    folds = [idx[0:4], idx[4:8], idx[8:12]]

    def _make_df() -> pd.DataFrame:
        # Categorical codes unique per row to make category tracking trivial
        cats = [f"C{i}" for i in idx]
        # Simple numeric feature and target correlated with index
        df = pd.DataFrame(
            {
                "cat": cats,
                "feat": idx.astype(float),
                "weight-(kg)": (idx * 0.5 + 1.0).astype(float),
                "split": ["train"] * len(idx),
            }
        )
        return df

    train_df = _make_df()

    # Monkeypatch data loader to return our in-memory splits
    monkeypatch.setattr(
        ridge_mod,
        "load_splits_from_parquet",
        lambda *_a, **_k: (train_df.drop(columns=["split"]), train_df.drop(columns=["split"]))
    )

    # Monkeypatch KFold to controlled splits
    class _KF:
        def __init__(self, *a: Any, **k: Any) -> None:  # noqa: D401
            pass

        def split(self, X: pd.DataFrame, *a: Any, **k: Any) -> Iterable[tuple[np.ndarray, np.ndarray]]:
            for i in range(n_folds):
                val = folds[i]
                train = np.setdiff1d(idx, val)
                yield train, val

    monkeypatch.setattr(ridge_mod, "KFold", _KF)

    # Record categories seen at each OneHotEncoder.fit
    fit_calls: list[set[str]] = []

    from sklearn.preprocessing import OneHotEncoder as SKOHE  # real class

    def _LoggedOHE(*args: Any, **kwargs: Any) -> SKOHE:
        real = SKOHE(*args, **kwargs)
        _orig_fit = real.fit

        def _fit(X: Any, y: Any | None = None) -> Any:
            arr = np.asarray(X)
            cats = set(str(v) for v in arr.ravel().tolist())
            fit_calls.append(cats)
            return _orig_fit(X, y)

        real.fit = _fit  # type: ignore[method-assign]
        return real

    monkeypatch.setattr(ridge_mod, "OneHotEncoder", _LoggedOHE)

    # Run optimization
    best_params, best_value, cv_summary = ridge_mod.optimize_ridge_hyperparameters(
        parquet_path=Path("dummy.parquet"),  # unused due to monkeypatch
        target_column="weight-(kg)",
        n_trials=1,
        random_state=123,
        cv_n_splits=n_folds,
    )

    # There should be exactly one OHE.fit per fold
    assert len(fit_calls) == n_folds

    # Check per-fold categories match the training subset categories
    for i in range(n_folds):
        val = folds[i]
        train = np.setdiff1d(idx, val)
        expected = set(str(c) for c in train_df.loc[train, "cat"].unique().tolist())
        assert fit_calls[i] == expected

    # Basic sanity on outputs
    assert "alpha" in best_params
    assert np.isfinite(best_value)
    assert "cv_mse_mean" in cv_summary
