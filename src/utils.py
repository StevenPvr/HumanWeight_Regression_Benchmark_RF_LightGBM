"""Project utilities centralizing reusable helpers."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.preprocessing import OneHotEncoder  # type: ignore

from src.constants import PROJECT_ROOT
from src.logging_setup import configure_logging

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
        configure_logging()
        _LOGGER_CONFIGURED = True
        # Fallback: ensure at least one stdout handler exists for tests/environments
        root_logger = logging.getLogger()
        if len(root_logger.handlers) == 0:
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(logging.INFO)
            root_logger.addHandler(sh)
    return logging.getLogger(name)


def save_splits_with_marker(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    csv_path: Path,
    parquet_path: Path,
) -> None:
    """
    Save train/test splits into single CSV and Parquet files with a split marker.

    WHY: A single file with a "split" column is convenient for downstream loading
    and analytics while keeping provenance of each row.

    Args:
        train_df: Training split DataFrame.
        test_df: Test split DataFrame.
        csv_path: Destination path for the concatenated CSV file.
        parquet_path: Destination path for the concatenated Parquet file.

    Raises:
        RuntimeError: If Parquet write fails due to missing engine.
    """
    # Build concatenated DataFrame without mutating inputs
    train_tagged = train_df.copy()
    train_tagged["split"] = "train"

    test_tagged = test_df.copy()
    test_tagged["split"] = "test"

    combined = pd.concat([train_tagged, test_tagged], ignore_index=True)

    # Ensure destination directories exist
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)

    for path in (csv_path, parquet_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    combined.to_csv(csv_path, index=False)

    # Write Parquet, surfacing a clear error if engine is missing
    try:
        combined.to_parquet(parquet_path, index=False)
    except Exception as exc:  # pragma: no cover - engine missing depends on environment
        raise RuntimeError("Failed to write Parquet. Install 'pyarrow' or 'fastparquet'.") from exc


def save_label_encoders_mappings(
    encoders: Mapping[str, OneHotEncoder],
    json_path: Path,
    csv_path: Path,
) -> None:
    """
    Persist OneHotEncoder mappings to JSON and CSV.

    - JSON contains categories and feature_names per column.
    - CSV uses encoded_value (feature name) for each category.
    """
    records: list[dict[str, str]] = []
    export: dict[str, dict[str, Any]] = {}

    for column_name, encoder in encoders.items():
        # OneHotEncoder path
        categories_raw = cast(Sequence[Any], getattr(encoder, "categories_", [[]]))
        categories = [str(x) for x in (categories_raw[0] if len(categories_raw) > 0 else [])]
        feature_names = encoder.get_feature_names_out([column_name]).tolist()
        mapping_ohe = {cat: fname for cat, fname in zip(categories, feature_names)}
        export[column_name] = {
            "categories": categories,
            "mapping": mapping_ohe,
            "feature_names": feature_names,
            "type": "OneHotEncoder",
        }
        for cat, fname in mapping_ohe.items():
            records.append(
                {
                    "column": column_name,
                    "original_value": cat,
                    "encoded_value": fname,
                }
            )

    # Ensure directories exist
    json_path = Path(json_path)
    csv_path = Path(csv_path)

    for path in (json_path, csv_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(export, handle, ensure_ascii=False, indent=2)

    # Write CSV (empty allowed)
    df_map = pd.DataFrame.from_records(
        records,
        columns=["column", "original_value", "encoded_value"],
    )
    df_map.to_csv(csv_path, index=False)


def load_splits_from_parquet(
    parquet_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train/test splits from a single Parquet file with a 'split' column.

    WHY: Downstream feature engineering and modeling need easy access to each split
    without leaking the split marker into features.

    Args:
        parquet_path: Path to the concatenated Parquet file produced by save_splits_with_marker.

    Returns:
        Tuple of DataFrames: (train_df, test_df), each without the 'split' column.

    Raises:
        FileNotFoundError: If the Parquet file does not exist.
        KeyError: If the 'split' column is missing in the file.
        RuntimeError: If Parquet read fails due to missing engine.
    """
    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:  # pragma: no cover - depends on environment engine
        raise RuntimeError("Failed to read Parquet. Install 'pyarrow' or 'fastparquet'.") from exc

    if "split" not in df.columns:
        raise KeyError("Missing 'split' column in the Parquet file.")

    def _subset(tag: str) -> pd.DataFrame:
        mask: pd.Series = df["split"] == tag
        subset = df.loc[mask].drop(columns=["split"], errors="ignore").reset_index(drop=True)
        return cast(pd.DataFrame, subset)

    train_df = _subset("train")
    test_df = _subset("test")
    # If a validation split is present, return a triple for callers expecting it
    if (df["split"] == "val").any():
        val_df = _subset("val")
        return train_df, val_df, test_df
    return train_df, test_df


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
            # WHY: Use pd.Categorical(...).codes to avoid Series.cat typing
            # issues with static analyzers
            df_out[col] = pd.Categorical(s).codes.astype("int32")
            continue

        # Object/string columns: prefer numeric coercion when it meaningfully succeeds,
        # otherwise fall back to stable categorical codes to avoid all-NaN columns.
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            try:
                numeric_guess = pd.to_numeric(s, errors="coerce")
            except Exception:
                numeric_guess = pd.Series([np.nan] * len(s), index=s.index)
            # Heuristic: if at least half the entries convert to numeric, keep them;
            # otherwise encode categories deterministically.
            non_na = int(numeric_guess.notna().sum())
            if non_na >= max(1, len(s) // 2):
                df_out[col] = numeric_guess
            else:
                df_out[col] = pd.Categorical(s.fillna("__MISSING__")).codes.astype("int32")
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
) -> tuple[list[pd.DataFrame], dict[str, OneHotEncoder]]:
    """Encode object/string columns with OneHotEncoder fitted on train only.

    Unseen categories in transformed frames are ignored (all-zero vector) via
    ``handle_unknown='ignore'``. Numeric columns are preserved unchanged.
    """

    categorical_columns = [
        column
        for column in train_df.columns
        if pd.api.types.is_object_dtype(train_df[column])
        or pd.api.types.is_string_dtype(train_df[column])
    ]

    encoders: dict[str, OneHotEncoder] = {}

    if not categorical_columns:
        outputs = [df.copy() for df in (train_df, *dataframes)]
        return outputs, encoders

    for column in categorical_columns:
        # WHY: Use drop='first' to avoid perfect multicollinearity and to make
        # unseen categories or the dropped baseline appear as all-zero vectors
        # within the group, which downstream tests assert explicitly.
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")
        # Reshape for sklearn (needs 2D array)
        col_data = train_df[[column]].astype("string").fillna("__MISSING__").values
        ohe.fit(col_data)
        encoders[column] = ohe

    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        transformed = df.copy()
        for column in categorical_columns:
            if column not in transformed.columns:
                continue
            ohe = encoders[column]
            col_data = transformed[[column]].astype("string").fillna("__MISSING__").values
            encoded_array = ohe.transform(col_data)
            feature_names = ohe.get_feature_names_out([column])
            encoded_df = pd.DataFrame(
                encoded_array,
                columns=feature_names,
                index=transformed.index,
            )
            transformed = transformed.drop(columns=[column])
            transformed = pd.concat([transformed, encoded_df], axis=1)
        return transformed

    encoded_train = _transform(train_df)
    encoded_others = [_transform(df) for df in dataframes]

    return [encoded_train, *encoded_others], encoders


def prepare_train_val_numeric_splits(
    parquet_path: Path,
    target_column: str,
) -> tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]:
    """Return numeric (X, y) for train and validation from a Parquet artifact.

    Accepts either a (train, val, test) triple or a (train, test) pair from
    ``load_splits_from_parquet``. When no explicit validation split is provided,
    the train split is further divided into train/val using a standard 80/20
    split without leakage.
    """
    loaded = load_splits_from_parquet(parquet_path)  # type: ignore[assignment]

    # Unpack 2- or 3-tuples (train, val, [test])
    if isinstance(loaded, tuple) and len(loaded) == 3:  # type: ignore[arg-type]
        train_df, val_df, _ = loaded  # type: ignore[misc]
    else:
        train_df, _test_df = cast(tuple[pd.DataFrame, pd.DataFrame], loaded)  # type: ignore[assignment]
        # Derive a validation split deterministically from train
        from src.data_preparation.data_preparation import split_train_test

        train_df, val_df = split_train_test(train_df, target_column=target_column, random_state=123)

    X_train_raw, y_train = split_features_target(train_df, target_column)
    X_val_raw, y_val = split_features_target(val_df, target_column)

    (X_train_enc, X_val_enc), _enc = fit_label_encoders_on_train(
        X_train_raw,
        [X_val_raw],
    )

    X_train_num = ensure_numeric_columns(X_train_enc)
    X_val_num = ensure_numeric_columns(X_val_enc)

    return (X_train_num, y_train), (X_val_num, y_val)


def prepare_train_numeric_splits(
    parquet_path: Path,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return encoded numeric train split from a Parquet artifact.

    WHY: Hyperparameter searches share the same preprocessing pipeline; a
    single helper keeps encoding rules consistent while preventing leakage.

    Args:
        parquet_path: Path to the concatenated Parquet file with a ``split`` column.
        target_column: Name of the regression target present in every split.

    Returns:
        Tuple ``(X_train, y_train)`` where the feature matrix is numeric
        and safe to pass into scikit-learn APIs.

    Raises:
        ValueError: If the target column leaks back into the feature matrices.
    """

    parquet_path = Path(parquet_path)
    loaded_splits = load_splits_from_parquet(parquet_path)
    if isinstance(loaded_splits, tuple) and len(loaded_splits) == 3:  # type: ignore[arg-type]
        train_df = loaded_splits[0]  # type: ignore[index]
    else:
        train_df, _ = cast(tuple[pd.DataFrame, pd.DataFrame], loaded_splits)  # type: ignore[assignment]

    X_train_raw, y_train = split_features_target(train_df, target_column)

    encoded_frames, _ = fit_label_encoders_on_train(
        X_train_raw,
        [],  # No validation frames
    )
    (X_train_encoded,) = encoded_frames

    X_train_numeric = ensure_numeric_columns(X_train_encoded)

    if target_column in X_train_numeric.columns:
        # WHY: immediate guard keeps assertions closer to the leakage source for faster debugging
        raise ValueError("Target column leaked into feature matrix after encoding.")

    return X_train_numeric, y_train


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


def read_best_params(params_json_path: Path) -> dict[str, Any]:
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
    params_json_path = Path(params_json_path)

    with params_json_path.open("r", encoding="utf-8") as file:
        payload: dict[str, Any] = json.load(file)

    best_params = payload.get("best_params")
    if not isinstance(best_params, dict):
        raise KeyError("JSON file must contain a 'best_params' object with mappings.")

    return best_params


def to_project_relative_path(path: str | Path) -> Path:
    """Return ``path`` relative to the repository root when possible.

    WHY: Persisting absolute paths makes artifacts machine-specific. Converting
    them to project-relative strings keeps results stable across environments
    while still resolving to an absolute location when the path lies outside the
    repository tree (e.g., temporary directories in tests).
    """

    resolved_path = Path(path).expanduser().resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT)
    except ValueError:
        return resolved_path


def save_training_results(results: dict[str, Any], json_path: Path) -> Path:
    """Persist training metrics and metadata to JSON.

    WHY: Storing a structured summary alongside the model artifact keeps
    downstream evaluation reproducible and auditability-friendly.

    Args:
        results: Mapping containing serialisable metrics or metadata.
        json_path: Destination file path. Parent directories are created if missing.

    Returns:
        Path to the saved JSON file relative to the project root when the file
        lives inside the repository, otherwise the absolute path.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("results payload cannot be empty")

    path_obj = Path(json_path).expanduser()
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    return to_project_relative_path(path_obj)


def _validate_id_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_columns: Sequence[str],
) -> None:
    """Validate that all id_columns exist in both DataFrames.

    Args:
        train_df: Training split DataFrame.
        test_df: Test split DataFrame.
        id_columns: List of column names to validate.

    Raises:
        KeyError: If any id_columns are missing from either DataFrame.
    """
    missing_train = [c for c in id_columns if c not in train_df.columns]
    missing_test = [c for c in id_columns if c not in test_df.columns]
    if missing_train or missing_test:
        missing = ", ".join(sorted(set(missing_train + missing_test)))
        raise KeyError(f"Missing id columns for leakage check: {missing}")


def _get_overlap_keys(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_columns: Sequence[str] | None,
) -> list[str] | None:
    """Determine which columns to use for overlap detection.

    Args:
        train_df: Training split DataFrame.
        test_df: Test split DataFrame.
        id_columns: Optional list of columns to use, or None to use common columns.

    Returns:
        List of column names to use for overlap detection, or None if no valid keys.
    """
    if id_columns is not None:
        return list(id_columns)
    # Fallback: compare exact-row overlaps over common columns
    keys = sorted(set(train_df.columns).intersection(set(test_df.columns)))
    return keys if keys else None


def _check_for_overlap(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    keys: list[str],
) -> None:
    """Check for overlapping entities and raise if found.

    Args:
        train_df: Training split DataFrame.
        test_df: Test split DataFrame.
        keys: List of column names to use for overlap detection.

    Raises:
        ValueError: If overlap is detected between train and test entities.
    """
    overlap = (
        train_df.loc[:, keys].merge(test_df.loc[:, keys], on=keys, how="inner").drop_duplicates()
    )
    if not overlap.empty:
        sample = overlap.head(5).to_dict(orient="records")
        raise ValueError(
            "Identity leakage detected between train and test. "
            f"Overlap count={len(overlap)} on keys={keys}. Sample={sample}"
        )


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
        _validate_id_columns(train_df, test_df, id_columns)

    keys = _get_overlap_keys(train_df, test_df, id_columns)
    if keys is None:
        return  # no shared columns; cannot compare

    _check_for_overlap(train_df, test_df, keys)


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
        second.extend(idx[n_first : n_first + n_second].tolist())

    return first, second
