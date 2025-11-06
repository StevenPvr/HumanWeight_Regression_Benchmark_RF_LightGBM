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
import warnings

import numpy as np
import pandas as pd
import pytest

# Suppress expected warnings from sklearn OneHotEncoder for unknown categories
warnings.filterwarnings(
    "ignore", message="Found unknown categories in columns", category=UserWarning
)

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


# Test helpers kept in conftest to avoid per-module complexity inflation
class _Axis:
    def set_visible(self, *_args, **_kwargs):  # pragma: no cover - safety
        return None


class _Fig:
    def __init__(self, tmp_dir: Path):
        self._tmp_dir = tmp_dir

    def savefig(self, path, dpi: int = 150):  # pragma: no cover - IO stub
        safe_dir = self._tmp_dir / "plots"
        safe_dir.mkdir(parents=True, exist_ok=True)
        (safe_dir / Path(str(path)).name).write_bytes(b"PNG")

    def tight_layout(self):  # pragma: no cover - no-op
        return None

    def add_subplot(self, *_args, **_kwargs):  # pragma: no cover - safety
        return _Axes(self)

    def get_axes(self):  # pragma: no cover - safety
        return []


class _Axes:
    def __init__(self, fig: _Fig):
        self._fig = fig

    def get_figure(self):  # pragma: no cover - safety
        return self._fig

    def get_yaxis(self):  # pragma: no cover - safety
        return _Axis()

    def get_xaxis(self):  # pragma: no cover - safety
        return _Axis()

    # Commonly used accessors in plots
    def set_title(self, *_args, **_kwargs):  # pragma: no cover - no-op
        return None

    def set_xlabel(self, *_args, **_kwargs):  # pragma: no cover - no-op
        return None

    def set_ylabel(self, *_args, **_kwargs):  # pragma: no cover - no-op
        return None

    def set_xticklabels(self, *_args, **_kwargs):  # pragma: no cover - no-op
        return None


class _PLT:
    def __init__(self, tmp_dir: Path):
        self._tmp_dir = tmp_dir

    def subplots(self, *args, **kwargs):  # pragma: no cover - minimal API
        fig = _Fig(self._tmp_dir)
        return fig, _Axes(fig)

    def close(self, *args, **_kwargs):  # pragma: no cover - no-op
        return None


@pytest.fixture
def plt_stub(tmp_path: Path) -> _PLT:
    """Provide a minimal Matplotlib-like object for monkeypatching plt.

    WHY: Pandas plot accessors expect an Axes/plt API. Returning a small stub
    keeps integration/E2E tests fast and independent from a full Matplotlib.
    """

    return _PLT(tmp_path)


@pytest.fixture
def pandas_plot_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``pandas.plotting.PlotAccessor`` a no-op to avoid Matplotlib.

    WHY: Some tests only need to assert pipeline orchestration, not visuals.
    Monkeypatching the accessor keeps those tests robust and light.
    """

    import pandas.plotting._core as _pd_plot_core  # type: ignore

    monkeypatch.setattr(_pd_plot_core.PlotAccessor, "__call__", lambda self, *a, **k: None)
