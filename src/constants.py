"""Project-wide constants.

WHY: Centralize shared configuration to keep modules simple and reduce
magic numbers and duplicated literals across the codebase (DRY + KISS).
"""

from __future__ import annotations

from pathlib import Path

# Root paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
PLOTS_DIR: Path = PROJECT_ROOT / "plots"
PERMUTATION_PLOTS_DIR: Path = PLOTS_DIR / "permutation"

# Relative paths (string-friendly) rooted at the repository
RELATIVE_DATA_DIR: Path = Path("data")
RELATIVE_RESULTS_DIR: Path = Path("results")
RELATIVE_PLOTS_DIR: Path = Path("plots")

# Default dataset/file names
DEFAULT_INPUT_CLEANED_FILE: str = "dataset_cleaned_final.csv"
DEFAULT_SPLITS_CSV_FILE: str = "dataset_splits_encoded.csv"
DEFAULT_SPLITS_PARQUET_FILE: str = "dataset_splits_encoded.parquet"
DEFAULT_ENCODERS_JSON_FILE: str = "encoders_mappings.json"
DEFAULT_ENCODERS_CSV_FILE: str = "encoders_mappings.csv"
DEFAULT_HYPEROPT_INPUT_PARQUET: Path = DATA_DIR / DEFAULT_SPLITS_PARQUET_FILE
DEFAULT_FEATURE_IMPORTANCE_JSON_PATH: Path = RESULTS_DIR / "feature_importances.json"
DEFAULT_FEATURE_IMPORTANCE_PLOT_PATH: Path = PERMUTATION_PLOTS_DIR / "feature_importances.png"

# ML-related defaults
TARGET_COLUMN: str = "weight-(kg)"
DEFAULT_RANDOM_STATE: int = 123
DEFAULT_TRAIN_FRAC: float = 0.6
DEFAULT_VAL_FRAC: float = 0.2
DEFAULT_TEST_FRAC: float = 0.2
DEFAULT_N_STRATA: int = 10

# Hyperparameter optimization defaults
HYPEROPT_DEFAULT_N_TRIALS: int = 100
HYPEROPT_LGBM_EARLY_STOPPING_ROUNDS: int = 50
HYPEROPT_LGBM_RESULTS_FILENAME: str = "best_lightgbm_params.json"
HYPEROPT_RF_RESULTS_FILENAME: str = "best_random_forest_params.json"
TARGET_COLUMN_CANDIDATES: tuple[str, ...] = (
    "weight-(kg)",
    "Weight_kg",
    "Weight (kg)",
    "weight_kg",
)
HYPEROPT_TARGET_COLUMN_CANDIDATES: tuple[str, ...] = TARGET_COLUMN_CANDIDATES

# Training defaults for CLI entry points
DEFAULT_TRAINING_PARQUET_PATH: Path = DATA_DIR / DEFAULT_SPLITS_PARQUET_FILE
DEFAULT_LIGHTGBM_PARAMS_PATH: Path = RESULTS_DIR / HYPEROPT_LGBM_RESULTS_FILENAME
DEFAULT_LIGHTGBM_MODEL_PATH: Path = RESULTS_DIR / "models" / "lightgbm"
DEFAULT_RANDOM_FOREST_PARAMS_PATH: Path = RESULTS_DIR / HYPEROPT_RF_RESULTS_FILENAME
DEFAULT_RANDOM_FOREST_MODEL_PATH: Path = RESULTS_DIR / "models" / "random_forest"

# Evaluation defaults for CLI entry points
DEFAULT_EVAL_RESULTS_DIR: Path = RESULTS_DIR / "eval"
DEFAULT_LIGHTGBM_MODEL_FILE: Path = DEFAULT_LIGHTGBM_MODEL_PATH.with_suffix(".joblib")
DEFAULT_RANDOM_FOREST_MODEL_FILE: Path = DEFAULT_RANDOM_FOREST_MODEL_PATH.with_suffix(".joblib")
DEFAULT_EVAL_BATCH_SIZE: int = 1024
DEFAULT_EVAL_N_JOBS: int = -1
DEFAULT_EVAL_MAX_DISPLAY: int = 20
DEFAULT_RANDOM_FOREST_SHAP_SAMPLE_SIZE: int = 512

