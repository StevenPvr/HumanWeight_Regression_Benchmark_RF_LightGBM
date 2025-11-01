from __future__ import annotations

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

import pytest

try:
    from .models import (
        create_random_forest_regressor,
        create_lightgbm_regressor,
    )
except ImportError:
    # Allow running this test file directly without package context
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.models.models import (
        create_random_forest_regressor,
        create_lightgbm_regressor,
    )


def test_create_random_forest_regressor_defaults() -> None:
    model = create_random_forest_regressor()
    assert isinstance(model, RandomForestRegressor)
    # Default seed must be deterministic
    assert getattr(model, "random_state") == 123


def test_create_random_forest_regressor_custom_seed() -> None:
    model = create_random_forest_regressor(random_state=999)
    assert getattr(model, "random_state") == 999


def test_create_lightgbm_regressor_defaults() -> None:
    model = create_lightgbm_regressor()
    assert isinstance(model, LGBMRegressor)
    assert getattr(model, "random_state") == 123


def test_create_lightgbm_regressor_custom_seed() -> None:
    model = create_lightgbm_regressor(random_state=123)
    assert getattr(model, "random_state") == 123


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
