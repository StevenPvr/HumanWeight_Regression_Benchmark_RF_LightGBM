from __future__ import annotations

from __future__ import annotations

import pytest
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

from src.models.models import (
    create_lightgbm_regressor,
    create_random_forest_regressor,
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
