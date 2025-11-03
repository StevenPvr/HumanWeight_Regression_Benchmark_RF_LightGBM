from __future__ import annotations

"""Shared pytest fixtures for the entire test suite.

The repository originally repeated path bootstrapping and random state
construction logic across several `test_*.py` modules.  Centralising them in a
single location keeps the individual tests focused on the specific behaviour
they validate while still providing deterministic inputs and predictable
imports.
"""

import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT: Path = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def _ensure_project_root_on_path() -> Iterator[None]:
    """Expose the project root to ``sys.path`` for absolute imports in tests."""

    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        yield
    finally:
        # We intentionally keep the project root on ``sys.path`` after the
        # fixture completes.  Removing it could break modules that keep global
        # references to imported packages between tests.
        pass


@pytest.fixture
def random_state_factory() -> Callable[[int | None], np.random.RandomState]:
    """Return a helper that builds ``RandomState`` instances on demand."""

    def _factory(seed: int | None = None) -> np.random.RandomState:
        return np.random.RandomState(0 if seed is None else seed)

    return _factory


@pytest.fixture
def tiny_weight_dataframe() -> pd.DataFrame:
    """Provide a compact synthetic dataset used across IO related tests."""

    return pd.DataFrame(
        {
            "age": [25, 30, 35],
            "weight-(kg)": [70.5, 68.1, 82.3],
        }
    )

