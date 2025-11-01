from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor


def create_random_forest_regressor(random_state: int = 123) -> RandomForestRegressor:
    """
    Create a RandomForestRegressor with a fixed random state for reproducibility.

    WHY: We now include RandomForest in the pipeline again; exposing a small
    factory keeps call sites concise and testable while centralizing defaults.

    Args:
        random_state: Seed to ensure deterministic behavior across runs.

    Returns:
        An unfitted RandomForestRegressor instance.
    """
    # Keep constructor minimal; HPO/training will override relevant params.
    return RandomForestRegressor(random_state=random_state)


def create_lightgbm_regressor(random_state: int = 123) -> LGBMRegressor:
    """
    Create a LightGBM regressor with a fixed random state for reproducibility.

    WHY: Experiments must be reproducible; hyperparameter tuning will adjust defaults later.

    Args:
        random_state: Seed to ensure deterministic behavior across runs.

    Returns:
        An unfitted LGBMRegressor instance.
    """
    return LGBMRegressor(random_state=random_state)
