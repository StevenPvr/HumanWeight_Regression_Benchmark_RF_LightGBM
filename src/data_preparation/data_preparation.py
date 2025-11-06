"""Data preparation helpers with deterministic, leak-safe behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.constants import DEFAULT_RANDOM_STATE, DEFAULT_TEST_FRAC


def encode_categorical_variables(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, OneHotEncoder]]:
    """Encode object/string columns with OneHotEncoder and return encoders.

    WHY: One-hot encoding preserves nominal information without imposing an
    arbitrary order like label encoding would. Unknown categories are handled
    safely as all-zero vectors.

    Args:
        df: Input DataFrame possibly containing categorical variables.

    Returns:
        - Encoded DataFrame where each categorical column is replaced by its
          one-hot columns named ``<col>_<category>``.
        - Dict mapping original column name to its fitted ``OneHotEncoder``.
    """
    encoded = df.copy()
    encoders: dict[str, OneHotEncoder] = {}

    # Detect categorical columns (object or string dtype)
    categorical_columns = [
        c
        for c in encoded.columns
        if pd.api.types.is_object_dtype(encoded[c]) or pd.api.types.is_string_dtype(encoded[c])
    ]

    if not categorical_columns:
        return encoded, encoders

    for col in categorical_columns:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=None)
        col_values = encoded[[col]].astype("string").fillna("__MISSING__").values
        ohe.fit(col_values)
        transformed = ohe.transform(col_values)
        feature_names = ohe.get_feature_names_out([col])
        encoded_df = pd.DataFrame(transformed, columns=feature_names, index=encoded.index)
        encoded = pd.concat([encoded.drop(columns=[col]), encoded_df], axis=1)
        encoders[col] = ohe

    return encoded, encoders


def shuffle_data(
    df: pd.DataFrame,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """
    Shuffle DataFrame rows in random order with reproducibility.

    Uses a random seed to ensure the shuffle is reproducible across runs.
    The number of rows is preserved, only the order changes.

    Args:
        df: Input DataFrame to shuffle
        random_state: Random seed for reproducibility.

    Returns:
        Shuffled DataFrame with reset index

    Example:
        >>> df_shuffled = shuffle_data(df, random_state=42)
    """
    # Shuffle using sample with frac=1 to get all rows in random order
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df_shuffled


def split_train_test(
    df: pd.DataFrame,
    target_column: str,
    test_frac: float = DEFAULT_TEST_FRAC,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproducible split into train/test only.

    WHY: We rely on cross-validation within the train split for model selection
    and early stopping. Keeping a single held-out test set avoids leakage and
    simplifies the pipeline.
    """
    # Validate inputs (val is removed; ensure fractions sum to 1)
    if target_column not in df.columns and not df.empty:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
    train_frac = float(1.0 - test_frac)
    total = float(train_frac + test_frac)
    if not np.isfinite(total) or not np.isclose(total, 1.0):
        raise ValueError("train_frac + test_frac must sum to 1.0")
    if train_frac < 0 or test_frac < 0:
        raise ValueError("Split fractions must be non-negative")

    if df.empty:
        return df.copy(), df.copy()

    rng = np.random.RandomState(random_state)

    # Hold out the test set without looking at the target (leakage guard)
    remaining_df, test_df = _holdout_test(df, test_frac=test_frac, rng=rng)

    # Everything else is train
    train_df = remaining_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df


def _holdout_test(
    df: pd.DataFrame,
    *,
    test_frac: float,
    rng: np.random.RandomState,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
