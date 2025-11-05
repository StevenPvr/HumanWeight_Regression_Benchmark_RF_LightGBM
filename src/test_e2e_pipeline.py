from __future__ import annotations

"""Lightweight E2E test focusing on orchestration order.

WHY: Keep the end-to-end check simple and fast. Detailed behaviour of each
pipeline stage is covered by targeted integration tests.
"""

from typing import Tuple

import pytest


def test_main_global_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate that main_global runs stages in the expected order."""

    from src import main_global

    calls: list[Tuple[str, tuple, tuple]] = []

    def _rec(label: str):
        def _fn(*args, **kwargs):  # pragma: no cover - trivial stub
            calls.append((label, args, tuple(sorted(kwargs.items()))))
            return 0

        return _fn

    monkeypatch.setattr(main_global, "data_cleaning_main", _rec("data_cleaning"))
    monkeypatch.setattr(main_global, "data_preparation_main", _rec("data_preparation"))
    monkeypatch.setattr(
        main_global, "hyperparameters_main", _rec("hyperparameters_optimization")
    )
    monkeypatch.setattr(main_global, "feature_engineering_main", _rec("feature_engineering"))
    monkeypatch.setattr(main_global, "training_main", _rec("training"))
    monkeypatch.setattr(main_global, "eval_main", _rec("eval"))
    monkeypatch.setattr(main_global, "baseline_main", _rec("baseline"))

    main_global.main()

    observed = [c[0] for c in calls]
    assert observed[:6] == [
        "data_cleaning",
        "data_preparation",
        "hyperparameters_optimization",
        "feature_engineering",
        "training",
        "eval",
    ]
    # Baseline called twice: optimize then train
    assert observed[-2:] == ["baseline", "baseline"]


if __name__ == "__main__":  # pragma: no cover - convenience
    pytest.main([__file__, "-v"])