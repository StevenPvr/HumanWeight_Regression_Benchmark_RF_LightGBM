"""Data preparation module for encoding categorical variables.

WHY: Keep simple, deterministic transforms and avoid leakage-prone patterns.
Refactors prefer copy-on-write and explicit seeding for reproducibility.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple, cast
import numpy as np
from src.constants import (
    DEFAULT_N_STRATA,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TRAIN_FRAC,
    DEFAULT_VAL_FRAC,
    DEFAULT_TEST_FRAC,
)
from src.utils import proportional_allocation_indices


def encode_categorical_variables(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Detect and encode categorical variables using LabelEncoder.

    This function automatically identifies columns with object dtype
    and applies LabelEncoder to transform them into numerical values.
    Compatible with tree-based models (Random Forest, XGBoost, LightGBM).

    Args:
        df: Input DataFrame with potential categorical variables

    Returns:
        Tuple containing:
            - DataFrame with encoded categorical variables
            - Dictionary mapping column names to their fitted LabelEncoders

    Example:
        >>> df_encoded, encoders = encode_categorical_variables(df)
    """
    df_encoded = df.copy()
    encoders = {}

    # Detect categorical columns (object dtype)
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns

    # Apply LabelEncoder to each categorical column
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le

    return df_encoded, encoders


def shuffle_data(df: pd.DataFrame, random_state: int = 123) -> pd.DataFrame:
    """
    Shuffle DataFrame rows in random order with reproducibility.

    Uses a random seed to ensure the shuffle is reproducible across runs.
    The number of rows is preserved, only the order changes.

    Args:
        df: Input DataFrame to shuffle
        random_state: Random seed for reproducibility (default: 123)

    Returns:
        Shuffled DataFrame with reset index

    Example:
        >>> df_shuffled = shuffle_data(df, random_state=42)
    """
    # Shuffle using sample with frac=1 to get all rows in random order
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df_shuffled


def split_train_val_test(
    df: pd.DataFrame,
    target_column: str,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_strata: int = DEFAULT_N_STRATA,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproducible split with no target-driven decisions using the test set.

    WHY: To avoid any test leakage, the test split is drawn first via a
    deterministic random permutation without looking at the target. The
    remaining data (train+val) is then stratified on the target using
    quantile bins to preserve distribution when splitting into train/val.

    For continuous targets, stratification is approximated via quantile bins
    on the train+val subset only. Within each stratum, items are apportioned
    to train/val so rounding produces at most ±1 difference vs ideal counts.
    """
    # WHY: Keep public function concise by delegating to smaller helpers that
    # each enforce a single responsibility (KISS). This also improves testability.
    _validate_split_inputs(df, target_column, train_frac, val_frac, test_frac)

    if df.empty:
        return df.copy(), df.copy(), df.copy()

    rng = np.random.RandomState(random_state)

    # 1) Hold out the test set without looking at the target (leakage guard)
    remaining_df, test_df = _holdout_test(df, test_frac=test_frac, rng=rng)

    # 2) Compute strata for the remaining data using the target only within train+val
    # WHY: Explicit cast helps static type checkers (Pyright) narrow df[col] to Series
    y_remaining = cast(pd.Series, remaining_df[target_column])
    strata_rem = _compute_strata_codes(y_remaining, n_strata=n_strata)

    # 3) Allocate train/val within each stratum using deterministic proportional rounding
    train_df, val_df = _allocate_train_val(
        remaining_df,
        strata=strata_rem,
        train_frac=train_frac,
        val_frac=val_frac,
        rng=rng,
    )

    return train_df, val_df, test_df


def _validate_split_inputs(
    df: pd.DataFrame,
    target_column: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> None:
    """Validate split inputs early to provide actionable errors.

    WHY: Failing fast with clear messages prevents silent misconfigurations
    (e.g., fractions not summing to 1) from propagating through the pipeline.
    """
    if target_column not in df.columns and not df.empty:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
    total = float(train_frac + val_frac + test_frac)
    if not np.isfinite(total) or not np.isclose(total, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")
    if train_frac < 0 or val_frac < 0 or test_frac < 0:
        raise ValueError("Split fractions must be non-negative")


def _holdout_test(
    df: pd.DataFrame,
    *,
    test_frac: float,
    rng: np.random.RandomState,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return remaining_df and test_df using a target-agnostic permutation.

    WHY: Drawing the test set first without using the target avoids any
    information leakage into hyperparameter tuning or feature selection.
    """
    n = len(df)
    if n == 0 or test_frac <= 0:
        return df.copy(), df.iloc[0:0].copy()
    permuted_idx = rng.permutation(df.index.to_numpy())
    n_test = int(round(test_frac * n))
    test_idx = permuted_idx[:n_test]
    remaining_idx = permuted_idx[n_test:]
    test_df = df.loc[test_idx].reset_index(drop=True)
    remaining_df = df.loc[remaining_idx]
    return remaining_df, test_df


def _compute_strata_codes(
    y: pd.Series,
    *,
    n_strata: int,
) -> pd.Series:
    """Compute integer strata codes for stratified allocation on remaining data.

    WHY: Approximating stratification for continuous targets with quantile bins
    preserves distribution without overfitting a small dataset.
    """
    if pd.api.types.is_numeric_dtype(y):
        num_bins = max(1, min(int(n_strata), int(y.nunique())))
        if num_bins == 1:
            return pd.Series(0, index=y.index)
        # WHY: Cast to Series after to_numeric so static analysis doesn't see a scalar/Unknown union
        y_num = pd.to_numeric(y, errors="coerce")
        y_float = cast(pd.Series, y_num).astype("float64")
        qbins = pd.qcut(y_float, q=num_bins, duplicates="drop")
        codes = pd.Categorical(qbins).codes  # -1 for NaN; acceptable as its own group
        return pd.Series(codes, index=y.index, dtype="int64")
    # Categorical/other types: use category codes directly
    return pd.Series(pd.Categorical(y).codes, index=y.index, dtype="int64")


def _allocate_train_val(
    remaining_df: pd.DataFrame,
    *,
    strata: pd.Series,
    train_frac: float,
    val_frac: float,
    rng: np.random.RandomState,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Allocate train/val within strata using deterministic proportional rounding.

    WHY: Splitting per-stratum with the same seeded RNG provides reproducible
    partitions while respecting desired ratios up to rounding.
    """
    total_tv = float(train_frac + val_frac)
    if total_tv <= 0:
        raise ValueError("train_frac + val_frac must be > 0")
    train_norm = float(train_frac / total_tv)
    val_norm = float(val_frac / total_tv)

    groups: dict[int, np.ndarray] = {}
    for gid, grp in remaining_df.assign(_s=strata).groupby("_s", sort=False):
        # WHY: Cast gid to int to satisfy type checker; strata is int64 Series from _compute_strata_codes
        gid_int = cast(int, gid)
        groups[gid_int] = grp.index.to_numpy()

    train_idx, val_idx = proportional_allocation_indices(
        groups, (train_norm, val_norm), rng=rng
    )

    train_df = remaining_df.loc[train_idx].reset_index(drop=True)
    val_df = remaining_df.loc[val_idx].reset_index(drop=True)
    return train_df, val_df
