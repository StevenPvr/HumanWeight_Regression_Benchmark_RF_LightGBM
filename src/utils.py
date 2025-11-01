"""Project utilities.

This module centralizes reusable helper functions.
"""

from __future__ import annotations

import logging
import os
import json
import sys
from typing import Dict, Sequence, Any, cast

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold

# WHY: local numpy import keeps numeric helpers close to consumers without polluting global namespace unnecessarily
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed, parallel
from contextlib import contextmanager


_LOGGER_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger configured once for the project.

    WHY: Centralizing logger creation ensures every module emits consistent,
    pipeline-friendly messages without duplicating ``basicConfig`` calls or
    accidentally stacking handlers.

    Args:
        name: Logical logger name, typically ``__name__`` of the caller.

    Returns:
        Logger instance ready for use.
    """

    global _LOGGER_CONFIGURED
    if not _LOGGER_CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.handlers = [handler]
        _LOGGER_CONFIGURED = True
    return logging.getLogger(name)


def save_splits_with_marker(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    csv_path: str,
    parquet_path: str,
) -> None:
    """
    Save train/val/test splits into single CSV and Parquet files with a split marker.

    WHY: A single file with a "split" column is convenient for downstream loading
    and analytics while keeping provenance of each row.

    Args:
        train_df: Training split DataFrame.
        val_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        csv_path: Destination path for the concatenated CSV file.
        parquet_path: Destination path for the concatenated Parquet file.

    Raises:
        RuntimeError: If Parquet write fails due to missing engine.
    """
    # Build concatenated DataFrame without mutating inputs
    train_tagged = train_df.copy()
    train_tagged["split"] = "train"

    val_tagged = val_df.copy()
    val_tagged["split"] = "val"

    test_tagged = test_df.copy()
    test_tagged["split"] = "test"

    combined = pd.concat([train_tagged, val_tagged, test_tagged], ignore_index=True)

    # Ensure destination directories exist
    for path in (csv_path, parquet_path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    # Write CSV
    combined.to_csv(csv_path, index=False)

    # Write Parquet, surfacing a clear error if engine is missing
    try:
        combined.to_parquet(parquet_path, index=False)
    except Exception as exc:  # pragma: no cover - engine missing depends on environment
        raise RuntimeError(
            "Failed to write Parquet. Install 'pyarrow' or 'fastparquet'."
        ) from exc


def save_label_encoders_mappings(
    encoders: Dict[str, LabelEncoder],
    json_path: str,
    csv_path: str,
) -> None:
    """
    Persist LabelEncoder mappings for categorical columns to JSON and CSV.

    JSON captures both the class ordering and explicit mapping. CSV provides a
    flat-table view with rows (column, original_value, encoded_value).
    """
    records = []
    export: Dict[str, dict] = {}

    for column_name, encoder in encoders.items():
        classes_raw = cast(Sequence[Any], getattr(encoder, "classes_", ()))
        classes = [str(x) for x in classes_raw]
        mapping = {cls: int(i) for i, cls in enumerate(classes)}
        export[column_name] = {
            "classes": classes,
            "mapping": mapping,
        }
        for cls, idx in mapping.items():
            records.append({
                "column": column_name,
                "original_value": cls,
                "encoded_value": idx,
            })

    # Ensure directories exist
    for path in (json_path, csv_path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    # Write JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    # Write CSV (empty allowed)
    df_map = pd.DataFrame.from_records(records, columns=[
        "column", "original_value", "encoded_value"
    ])
    df_map.to_csv(csv_path, index=False)



def load_splits_from_parquet(
    parquet_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train/val/test splits from a single Parquet file with a 'split' column.

    WHY: Downstream feature engineering and modeling need easy access to each split
    without leaking the split marker into features.

    Args:
        parquet_path: Path to the concatenated Parquet file produced by save_splits_with_marker.

    Returns:
        Tuple of DataFrames: (train_df, val_df, test_df), each without the 'split' column.

    Raises:
        FileNotFoundError: If the Parquet file does not exist.
        KeyError: If the 'split' column is missing in the file.
        RuntimeError: If Parquet read fails due to missing engine.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:  # pragma: no cover - depends on environment engine
        raise RuntimeError(
            "Failed to read Parquet. Install 'pyarrow' or 'fastparquet'."
        ) from exc

    if "split" not in df.columns:
        raise KeyError("Missing 'split' column in the Parquet file.")

    def _subset(tag: str) -> pd.DataFrame:
        mask: pd.Series = df["split"] == tag
        subset = df.loc[mask].drop(columns=["split"], errors="ignore").reset_index(drop=True)
        return cast(pd.DataFrame, subset)

    train_df = _subset("train")
    val_df = _subset("val")
    test_df = _subset("test")

    return train_df, val_df, test_df


def ensure_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the DataFrame where every column is numeric.

    WHY: Tree-based models and permutation importance require numeric inputs.
    Strategy:
      - Numeric columns are left unchanged
      - Booleans are cast to int64
      - Datetimes/timedeltas are converted to int64 (ns)
      - Categoricals are replaced by their integer codes (int32)
      - Object/string columns are coerced with pd.to_numeric(errors="coerce")
    """
    df_out = df.copy()

    for col in df_out.columns:
        # WHY: Explicit cast helps static analyzers (Pyright) understand Series methods
        s: pd.Series = cast(pd.Series, df_out[col])
        if pd.api.types.is_numeric_dtype(s):
            continue
        if pd.api.types.is_bool_dtype(s):
            df_out[col] = s.astype("int64")
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            df_out[col] = s.astype("int64")
            continue
        if pd.api.types.is_timedelta64_dtype(s):
            df_out[col] = s.astype("int64")
            continue
        if isinstance(s.dtype, pd.CategoricalDtype):
            # WHY: Use pd.Categorical(...).codes to avoid Series.cat typing issues with static analyzers
            df_out[col] = pd.Categorical(s).codes.astype("int32")
            continue

        # Fallback for object/string/other types
        try:
            df_out[col] = pd.to_numeric(s, errors="coerce")
        except Exception:
            # WHY: Some exotic dtypes require a string round-trip before coercion
            df_out[col] = pd.to_numeric(s.astype("string"), errors="coerce")

    return df_out


def round_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    decimals: int = 1,
) -> pd.DataFrame:
    """
    Return a copy of df with selected columns rounded to a fixed precision.

    WHY: Keep engineered metrics (e.g., BMI, fat %) consistent and readable by
    rounding to a stable number of decimals without mutating the caller.

    Args:
        df: Input DataFrame.
        columns: Column names to round if present.
        decimals: Number of decimal places (default: 1).

    Returns:
        A new DataFrame with specified columns rounded when applicable.

    Notes:
        - Missing columns are ignored (KISS + robust to partial frames).
        - Non-numeric columns are coerced with pd.to_numeric before rounding.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue  # tolerate absent engineered columns
        # WHY: Coerce to numeric first to ensure stable rounding semantics
        series_numeric_s: pd.Series = cast(pd.Series, pd.to_numeric(out[col], errors="coerce"))
        dec = int(decimals)
        # WHY: Use Python's round to avoid pandas/numpy overload ambiguity (freq vs decimals)
        values = series_numeric_s.tolist()
        rounded_list: list[float | None] = [
            (None if pd.isna(v) else round(float(v), dec)) for v in values  # type: ignore[arg-type]
        ]
        out[col] = pd.Series(rounded_list, index=out.index, dtype="object")
    return out

def fit_label_encoders_on_train(
    train_df: pd.DataFrame,
    dataframes: Sequence[pd.DataFrame],
) -> tuple[list[pd.DataFrame], dict[str, LabelEncoder]]:
    """Encode object/string columns using train split statistics only..

    Args:
        train_df: DataFrame used to fit the categorical encoders.
        dataframes: Sequence of additional DataFrames to transform using the
            fitted encoders. Ordering is preserved in the returned list.

    Returns:
        Tuple containing:
            - List of transformed DataFrames starting with the encoded
              ``train_df`` followed by the transformed ``dataframes``.
            - Dictionary mapping column names to their fitted ``LabelEncoder``.
    """

    categorical_columns = [
        column
        for column in train_df.columns
        if pd.api.types.is_object_dtype(train_df[column])
        or pd.api.types.is_string_dtype(train_df[column])
    ]

    encoders: dict[str, LabelEncoder] = {}

    if not categorical_columns:
        outputs = [df.copy() for df in (train_df, *dataframes)]
        return outputs, encoders

    missing_token = "__MISSING__"

    def _sanitize(series: pd.Series) -> pd.Series:
        # WHY: Normalize to string dtype with an explicit missing token to keep mapping stable
        sanitized = series.astype("string").fillna(missing_token)
        return sanitized.astype(str)

    for column in categorical_columns:
        encoder = LabelEncoder()
        # WHY: Cast helps Pyright narrow index access (Series | DataFrame)
        sanitized_train = _sanitize(cast(pd.Series, train_df[column]))
        # WHY: Use tolist() to avoid pandas .to_numpy typing ambiguity
        encoder.fit(sanitized_train.tolist())
        encoders[column] = encoder

    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        transformed = df.copy()
        for column in categorical_columns:
            if column not in transformed.columns:
                continue
            sanitized_series = _sanitize(cast(pd.Series, transformed[column]))
            encoder = encoders[column]
            # WHY: After fit, classes_ is array-like; cast for static type checkers
            classes_any: Any = getattr(encoder, "classes_", [])
            classes_seq: Sequence[str] = cast(Sequence[str], list(classes_any))
            mapping = {cls: int(idx) for idx, cls in enumerate(classes_seq)}
            mapped = sanitized_series.map(mapping)
            filled = cast(pd.Series, mapped).fillna(-1)
            transformed[column] = filled.astype("int64")
        return transformed

    encoded_train = _transform(train_df)
    encoded_others = [_transform(df) for df in dataframes]

    return [encoded_train, *encoded_others], encoders


def prepare_train_val_numeric_splits(
    parquet_path: str,
    target_column: str,
) -> tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]:
    """Return encoded numeric train/validation splits from a Parquet artifact.

    WHY: Hyperparameter searches share the same preprocessing pipeline; a
    single helper keeps encoding rules consistent while preventing leakage.

    Args:
        parquet_path: Path to the concatenated Parquet file with a ``split`` column.
        target_column: Name of the regression target present in every split.

    Returns:
        Tuple ``((X_train, y_train), (X_val, y_val))`` where both feature
        matrices are numeric copies safe to pass into scikit-learn APIs.

    Raises:
        ValueError: If the target column leaks back into the feature matrices.
    """

    train_df, val_df, _ = load_splits_from_parquet(parquet_path)

    X_train_raw, y_train = split_features_target(train_df, target_column)
    X_val_raw, y_val = split_features_target(val_df, target_column)

    encoded_frames, _ = fit_label_encoders_on_train(
        X_train_raw,
        [X_val_raw],
    )
    X_train_encoded, X_val_encoded = encoded_frames

    X_train_numeric = ensure_numeric_columns(X_train_encoded)
    X_val_numeric = ensure_numeric_columns(X_val_encoded)

    if target_column in X_train_numeric.columns or target_column in X_val_numeric.columns:
        # WHY: immediate guard keeps assertions closer to the leakage source for faster debugging
        raise ValueError("Target column leaked into feature matrix after encoding.")

    return (X_train_numeric, y_train), (X_val_numeric, y_val)


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a DataFrame into features X and target y.

    Args:
        df: Input DataFrame containing features and target.
        target_column: Name of the target column to extract.

    Returns:
        Tuple (X, y) where X excludes the target column and y is the target series.

    Raises:
        ValueError: If the target_column does not exist in df.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    X = df.drop(columns=[target_column]).copy()
    target = df.loc[:, target_column]
    if isinstance(target, pd.DataFrame):
        target_df = cast(pd.DataFrame, target)
        target = target_df.iloc[:, 0]
    y_series = cast(pd.Series, target)
    return X, y_series.copy()


def read_best_params(params_json_path: str) -> dict[str, Any]:
    """Return hyperparameters stored under the 'best_params' key of a JSON file.

    WHY: Centralize validation of tuning outputs so training code stays focused
    on model fitting while surfacing consistent errors for malformed artifacts.

    Args:
        params_json_path: Absolute or relative path to the JSON file produced by
            the hyperparameter search pipeline.

    Returns:
        Dictionary containing the tuned hyperparameters.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        KeyError: If the expected 'best_params' object is absent or invalid.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    import json

    with open(params_json_path, "r", encoding="utf-8") as file:
        payload: dict[str, Any] = json.load(file)

    best_params = payload.get("best_params")
    if not isinstance(best_params, dict):
        raise KeyError("JSON file must contain a 'best_params' object with mappings.")

    return best_params


def save_training_results(results: dict[str, Any], json_path: str) -> str:
    """Persist training metrics and metadata to JSON.

    WHY: Storing a structured summary alongside the model artifact keeps
    downstream evaluation reproducible and auditability-friendly.

    Args:
        results: Mapping containing serialisable metrics or metadata.
        json_path: Destination file path. Parent directories are created if missing.

    Returns:
        Absolute path to the saved JSON file.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("results payload cannot be empty")

    directory = os.path.dirname(json_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    return os.path.abspath(json_path)


def assert_no_overlap_between_train_and_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    id_columns: Sequence[str] | None = None,
) -> None:
    """Raise if the same entity appears in both train and test.

    WHY: Ensure there is no identity leakage (e.g., the same person present in
    both train and test). When explicit identifiers are known, pass them via
    ``id_columns``. Otherwise, we conservatively fall back to checking for exact
    duplicate rows across the intersection of columns, which is a safe baseline
    but may miss entity duplicates if rows differ.

    Args:
        train_df: Training split DataFrame.
        test_df: Test split DataFrame.
        id_columns: Optional list of columns that uniquely identify an entity
            (e.g., ["person_id"]). When provided, overlap is computed on these
            columns. When omitted, overlap is computed on the full intersection
            of columns between the two splits.

    Raises:
        KeyError: If any of the ``id_columns`` are missing from either split.
        ValueError: If overlap is detected between train and test entities.
    """
    if train_df.empty or test_df.empty:
        return  # nothing to compare

    if id_columns is not None:
        missing_train = [c for c in id_columns if c not in train_df.columns]
        missing_test = [c for c in id_columns if c not in test_df.columns]
        if missing_train or missing_test:
            missing = ", ".join(sorted(set(missing_train + missing_test)))
            raise KeyError(f"Missing id columns for leakage check: {missing}")
        keys = list(id_columns)
    else:
        # Fallback: compare exact-row overlaps over common columns
        keys = sorted(set(train_df.columns).intersection(set(test_df.columns)))
        if not keys:
            return  # no shared columns; cannot compare

    # Merge on keys to find overlapping entities/rows. Drop duplicates to keep the error concise.
    overlap = (
        train_df.loc[:, keys]
        .merge(test_df.loc[:, keys], on=keys, how="inner")
        .drop_duplicates()
    )

    if not overlap.empty:
        # Provide a compact preview to aid debugging without dumping full data
        sample = overlap.head(5).to_dict(orient="records")
        raise ValueError(
            "Identity leakage detected between train and test. "
            f"Overlap count={len(overlap)} on keys={keys}. Sample={sample}"
        )


def proportional_allocation_indices(
    groups: dict[int, np.ndarray],
    proportions: tuple[float, float],
    *,
    rng: np.random.RandomState,
) -> tuple[list[int], list[int]]:
    """Split grouped indices into two parts with proportional rounding.

    WHY: When stratifying within groups (e.g., target quantile bins), we want
    deterministic allocation that exactly matches the requested proportions up
    to rounding per group. We floor the ideal counts and distribute the leftover
    one-by-one to the splits with the largest fractional parts. This mirrors the
    behavior asserted in tests and avoids sklearn implementation subtleties.

    Args:
        groups: Mapping of group id -> 1D array of row indices belonging to it.
        proportions: Desired fractions for the two splits (a, b); only the
            relative ratio matters and is normalized internally.
        rng: Pre-seeded ``np.random.RandomState`` used to shuffle within groups
            for reproducibility.

    Returns:
        Tuple of lists (first_split_indices, second_split_indices).
    """
    a, b = proportions
    total = float(a + b)
    if total <= 0:
        raise ValueError("Sum of proportions must be > 0")
    a /= total
    b /= total

    first: list[int] = []
    second: list[int] = []

    for _gid, idx in groups.items():
        if idx.size == 0:
            continue
        idx = idx.copy()
        rng.shuffle(idx)  # reproducible randomness within the group

        m = int(idx.size)
        ideal = np.array([a * m, b * m], dtype=float)
        base = ideal.astype(int)
        remainder = int(m - base.sum())
        if remainder > 0:
            frac = ideal - base
            order = np.argsort(-frac)  # largest fractional parts first
            base[order[:remainder]] += 1
        n_first, n_second = int(base[0]), int(base[1])
        # Guard against numerical corner cases
        n_first = max(0, min(n_first, m))
        n_second = max(0, min(n_second, m - n_first))

        first.extend(idx[:n_first].tolist())
        second.extend(idx[n_first:n_first + n_second].tolist())

    return first, second
