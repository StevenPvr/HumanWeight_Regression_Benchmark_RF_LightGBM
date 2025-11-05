"""Project-wide constants.

WHY: Centralize shared configuration to keep modules simple and reduce
magic numbers and duplicated literals across the codebase (DRY + KISS).
"""

from __future__ import annotations

from pathlib import Path

# Root paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
PLOTS_DIR: Path = PROJECT_ROOT / "plots"
PERMUTATION_PLOTS_DIR: Path = PLOTS_DIR / "permutation"
SHAP_PLOTS_DIR: Path = PLOTS_DIR / "shape"

# Relative paths (string-friendly) rooted at the repository
RELATIVE_DATA_DIR: Path = Path("data")
RELATIVE_RESULTS_DIR: Path = Path("results")
RELATIVE_PLOTS_DIR: Path = Path("plots")
RELATIVE_SHAP_PLOTS_DIR: Path = RELATIVE_PLOTS_DIR / "shape"

# Logging
PIPELINE_LOG_PATH: Path = RESULTS_DIR / "pipeline.log"

# Default dataset/file names
DEFAULT_INPUT_CLEANED_FILE: str = "dataset_cleaned_final.csv"
DEFAULT_SPLITS_CSV_FILE: str = "dataset_splits_encoded.csv"
DEFAULT_SPLITS_PARQUET_FILE: str = "dataset_splits_encoded.parquet"
DEFAULT_ENCODERS_JSON_FILE: str = "encoders_mappings.json"
DEFAULT_ENCODERS_CSV_FILE: str = "encoders_mappings.csv"
DEFAULT_STANDARD_SCALED_CSV_FILE: str = "dataset_standard_scaled.csv"
DEFAULT_STANDARD_SCALED_PARQUET_FILE: str = "dataset_standard_scaled.parquet"
DEFAULT_HYPEROPT_INPUT_PARQUET: Path = DATA_DIR / DEFAULT_SPLITS_PARQUET_FILE
DEFAULT_FEATURE_IMPORTANCE_JSON_PATH: Path = RESULTS_DIR / "feature_importances.json"
DEFAULT_FEATURE_IMPORTANCE_PLOT_PATH: Path = PERMUTATION_PLOTS_DIR / "feature_importances.png"

# ML-related defaults
TARGET_COLUMN: str = "weight-(kg)"
DEFAULT_RANDOM_STATE: int = 123
DEFAULT_TRAIN_FRAC: float = 0.8
DEFAULT_TEST_FRAC: float = 0.2

# Hyperparameter optimization defaults
HYPEROPT_DEFAULT_N_TRIALS: int = 2
HYPEROPT_LGBM_RESULTS_FILENAME: str = "best_lightgbm_params.json"
HYPEROPT_RF_RESULTS_FILENAME: str = "best_random_forest_params.json"
HYPEROPT_RIDGE_RESULTS_FILENAME: str = "best_ridge_params.json"
TARGET_COLUMN_CANDIDATES: tuple[str, ...] = (
    "weight-(kg)",
    "Weight_kg",
    "Weight (kg)",
    "weight_kg",
)
HYPEROPT_TARGET_COLUMN_CANDIDATES: tuple[str, ...] = TARGET_COLUMN_CANDIDATES

# Nombre de folds utilisé pour la CV classique (K-fold).
CV_N_SPLITS = 5

# Training defaults for CLI entry points
DEFAULT_TRAINING_PARQUET_PATH: Path = DATA_DIR / DEFAULT_SPLITS_PARQUET_FILE
DEFAULT_LIGHTGBM_PARAMS_PATH: Path = RESULTS_DIR / HYPEROPT_LGBM_RESULTS_FILENAME
DEFAULT_LIGHTGBM_MODEL_PATH: Path = RESULTS_DIR / "models" / "lightgbm"
DEFAULT_RANDOM_FOREST_PARAMS_PATH: Path = RESULTS_DIR / HYPEROPT_RF_RESULTS_FILENAME
DEFAULT_RANDOM_FOREST_MODEL_PATH: Path = RESULTS_DIR / "models" / "random_forest"
DEFAULT_RIDGE_PARAMS_PATH: Path = RESULTS_DIR / HYPEROPT_RIDGE_RESULTS_FILENAME
DEFAULT_RIDGE_MODEL_PATH: Path = RESULTS_DIR / "models" / "ridge"
DEFAULT_RIDGE_SCALER_PATH: Path = RESULTS_DIR / "models" / "ridge_scaler.joblib"

# Evaluation defaults for CLI entry points
DEFAULT_EVAL_RESULTS_DIR: Path = RESULTS_DIR / "eval"
DEFAULT_LIGHTGBM_MODEL_FILE: Path = DEFAULT_LIGHTGBM_MODEL_PATH.with_suffix(".joblib")
DEFAULT_RANDOM_FOREST_MODEL_FILE: Path = DEFAULT_RANDOM_FOREST_MODEL_PATH.with_suffix(".joblib")
DEFAULT_RIDGE_MODEL_FILE: Path = DEFAULT_RIDGE_MODEL_PATH.with_suffix(".joblib")
DEFAULT_EVAL_BATCH_SIZE: int = 1024
DEFAULT_EVAL_N_JOBS: int = -1
DEFAULT_EVAL_MAX_DISPLAY: int = 20

# Feature engineering defaults
DEFAULT_PERMUTATION_REPEATS: int = 10
DEFAULT_PERMUTATION_N_JOBS: int = -1

# SHAP output structure
LIGHTGBM_SHAP_DIR: Path = SHAP_PLOTS_DIR / "LightGBM"
RANDOM_FOREST_SHAP_DIR: Path = SHAP_PLOTS_DIR / "rf"
