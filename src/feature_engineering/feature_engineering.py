"""Feature engineering utilities.

Contains helpers to compute, persist, and visualize permutation feature
importances with a RandomForest baseline.
"""

from __future__ import annotations

import json
import os
from typing import cast

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.utils import Bunch

from src.constants import DEFAULT_RANDOM_STATE
from src.models.models import create_random_forest_regressor
from src.utils import ensure_numeric_columns


def compute_random_forest_permutation_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    *,
    n_repeats: int = 50,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Compute permutation feature importance using a RandomForest regressor.

    WHY: Fit the model on the training split and evaluate permutation impact on
    the validation split to avoid data leakage while estimating feature effects.

    Args:
        X_train: Training features DataFrame.
        y_train: Training target Series.
        X_eval: Evaluation features DataFrame used for permutation.
        y_eval: Evaluation target Series.
        n_repeats: Number of shuffles per feature (default: 10).
        random_state: Random seed for reproducibility.
        n_jobs: Parallel jobs for permutation computation (default: -1).

    Returns:
        DataFrame with columns ["feature", "importance_mean", "importance_std"],
        sorted by descending importance.
    """
    # Ensure purely numeric matrices for stable model interaction
    X_train_num = ensure_numeric_columns(X_train)
    X_eval_num = ensure_numeric_columns(X_eval)

    # Fit RandomForest on training data only (no leakage from validation)
    rf = create_random_forest_regressor(random_state=int(random_state))
    rf.fit(X_train_num, y_train)

    # Use R^2 by default (estimator.score) to avoid sign ambiguity of neg-MSE
    result = cast(
        Bunch,
        permutation_importance(
            rf,
            X_eval_num,
            y_eval,
            n_repeats=int(n_repeats),
            random_state=int(random_state),
            n_jobs=int(n_jobs),
        ),
    )

    importance_df = pd.DataFrame(
        {
            "feature": list(X_eval_num.columns),
            "importance_mean": result.importances_mean.astype(float),
            "importance_std": result.importances_std.astype(float),
        }
    )
    importance_df = cast(
        pd.DataFrame,
        importance_df.sort_values("importance_mean", ascending=False).reset_index(drop=True),
    )
    return importance_df


def save_feature_importances_to_json(
    importance_df: pd.DataFrame,
    json_path: str,
) -> None:
    """
    Save permutation feature importances to a JSON file.

    WHY: Persisting results enables reproducible reporting and downstream analysis.

    Expected columns in importance_df: 'feature', 'importance_mean', 'importance_std'.
    The JSON will be a list of objects, one per feature.
    """
    required_cols = {"feature", "importance_mean", "importance_std"}
    if not required_cols.issubset(set(importance_df.columns)):
        missing = required_cols.difference(set(importance_df.columns))
        raise ValueError(f"importance_df is missing columns: {sorted(missing)}")

    # Ensure destination directory exists

    directory = os.path.dirname(json_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Convert to plain Python types for JSON serialization
    records = []
    for _, row in importance_df.iterrows():
        records.append(
            {
                "feature": str(row["feature"]),
                "importance_mean": float(row["importance_mean"]),
                "importance_std": float(row["importance_std"]),
            }
        )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def plot_feature_importances(
    importance_df: pd.DataFrame,
    output_path: str,
    *,
    top_n: int | None = None,
    title: str | None = "Permutation Feature Importance",
) -> None:
    """
    Create a horizontal bar plot of permutation feature importances.

    Args:
        importance_df: DataFrame with columns 'feature', 'importance_mean', 'importance_std'.
        output_path: Path to save the image (e.g., PNG).
        top_n: If set, limit to the top N features by mean importance.
        title: Optional plot title.

    Raises:
        ValueError: If required columns are missing.
        RuntimeError: If matplotlib is unavailable.
    """
    required_cols = {"feature", "importance_mean", "importance_std"}
    if not required_cols.issubset(set(importance_df.columns)):
        missing = required_cols.difference(set(importance_df.columns))
        raise ValueError(f"importance_df is missing columns: {sorted(missing)}")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting. Install 'matplotlib'.") from exc

    # Prepare data
    df_plot = importance_df.copy()
    if top_n is not None:
        df_plot = df_plot.nlargest(int(top_n), "importance_mean")
    df_plot = df_plot.sort_values("importance_mean", ascending=True)

    # Ensure dir exists
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Adaptive figure height for readability
    height = max(4.0, min(16.0, 1.0 + 0.4 * len(df_plot)))
    fig, ax = plt.subplots(figsize=(10.0, height))

    ax.barh(
        df_plot["feature"],
        df_plot["importance_mean"].astype(float),
        xerr=df_plot["importance_std"].astype(float),
        color="#4C78A8",
        ecolor="#999999",
        alpha=0.9,
        linewidth=0.5,
    )
    ax.set_xlabel("Mean decrease in score (permutation)")
    ax.set_ylabel("Feature")
    if title:
        ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
