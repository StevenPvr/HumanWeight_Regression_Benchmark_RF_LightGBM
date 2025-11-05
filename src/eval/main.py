"""CLI helpers to evaluate trained models on the held-out test split."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.constants import (  # noqa: E402
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_EVAL_MAX_DISPLAY,
    DEFAULT_EVAL_N_JOBS,
    DEFAULT_EVAL_RESULTS_DIR,
    DEFAULT_LIGHTGBM_MODEL_FILE,
    DEFAULT_RANDOM_FOREST_MODEL_FILE,
    DEFAULT_RIDGE_MODEL_FILE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TRAINING_PARQUET_PATH,
    LIGHTGBM_SHAP_DIR,
    RANDOM_FOREST_SHAP_DIR,
    DEFAULT_RIDGE_SCALER_PATH,
    TARGET_COLUMN,
)
from src.eval.eval import evaluate_lightgbm_on_test, evaluate_ridge_on_test  # noqa: E402
from src.utils import (  # noqa: E402
    ensure_numeric_columns,
    get_logger,
    load_splits_from_parquet,
    save_training_results,
    to_project_relative_path,
)

# WHY: Re-export utility helpers so integration tests can monkeypatch them here.
__all__ = [
    "build_arg_parser",
    "determine_model_selection",
    "evaluate_model_and_record",
    "execute_selected_models",
    "main",
    "ensure_numeric_columns",
    "load_splits_from_parquet",
    "save_training_results",
]

LOGGER = get_logger(__name__)

SHAP_DIR_BY_MODEL = {
    "lightgbm": LIGHTGBM_SHAP_DIR,
    "random_forest": RANDOM_FOREST_SHAP_DIR,
}


def build_arg_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser for evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate trained models on the held-out test split.",
    )
    parser.add_argument(
        "--parquet",
        default=str(DEFAULT_TRAINING_PARQUET_PATH),
        help=(
            "Path to the concatenated Parquet dataset with a 'split' column. "
            f"Default: {DEFAULT_TRAINING_PARQUET_PATH}"
        ),
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_LIGHTGBM_MODEL_FILE),
        help=(
            "Path to the saved LightGBM model artefact (joblib/pkl). "
            f"Default: {DEFAULT_LIGHTGBM_MODEL_FILE}"
        ),
    )
    parser.add_argument(
        "--rf-model",
        default=str(DEFAULT_RANDOM_FOREST_MODEL_FILE),
        help=(
            "Path to the saved RandomForest model artefact. "
            f"Default: {DEFAULT_RANDOM_FOREST_MODEL_FILE}"
        ),
    )
    parser.add_argument(
        "--ridge-model",
        default=str(DEFAULT_RIDGE_MODEL_FILE),
        help=(
            "Path to the saved Ridge model artefact (Pipeline joblib). "
            f"Default: {DEFAULT_RIDGE_MODEL_FILE}"
        ),
    )
    parser.add_argument(
        "--ridge-scaler",
        default=str(DEFAULT_RIDGE_SCALER_PATH),
        help=(
            "Path to the saved Ridge scaler artefact (kept for compatibility). "
            f"Default: {DEFAULT_RIDGE_SCALER_PATH}"
        ),
    )
    parser.add_argument(
        "--target-column",
        default=TARGET_COLUMN,
        help=f"Name of the regression target column (default: {TARGET_COLUMN}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
        help=f"Batch size used by downstream tooling (default: {DEFAULT_EVAL_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_EVAL_N_JOBS,
        help=f"Number of parallel jobs for optional post-processing (default: {DEFAULT_EVAL_N_JOBS}).",
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=DEFAULT_EVAL_MAX_DISPLAY,
        help=f"Maximum rows to display in reports (default: {DEFAULT_EVAL_MAX_DISPLAY}).",
    )
    # Accept both --eval-dir and --output-dir for compatibility with tests and docs.
    # WHY: The E2E pipeline invokes --output-dir while unit tests use --eval-dir.
    # Using two option strings mapped to the same destination keeps the CLI simple
    # and avoids breaking existing callers.
    parser.add_argument(
        "--eval-dir",
        "--output-dir",
        dest="eval_dir",
        default=str(DEFAULT_EVAL_RESULTS_DIR),
        help=(
            "Directory where evaluation summaries are stored (alias: --output-dir). "
            f"Default: {DEFAULT_EVAL_RESULTS_DIR}."
        ),
    )
    parser.add_argument(
        "--models",
        choices=["lightgbm", "random_forest", "ridge", "both", "all"],
        default=None,
        help=(
            "Which model(s) to evaluate. When omitted, an interactive prompt "
            "selects between LightGBM, RandomForest, Ridge, or all."
        ),
    )
    return parser


def determine_model_selection(args: argparse.Namespace) -> str:
    """Return the model selection based on CLI arguments and input prompts."""

    if args.models is not None:
        return str(args.models)

    try:
        if sys.stdin.isatty():
            choice = input(
                "Select models to evaluate: [1] LightGBM, [2] RandomForest, [3] Ridge, [4] All (default 4): "
            ).strip()
            if choice == "1":
                return "lightgbm"
            if choice == "2":
                return "random_forest"
            if choice == "3":
                return "ridge"
    except Exception:
        LOGGER.debug("Interactive selection failed; defaulting to all models")

    return "all"


def evaluate_model_and_record(
    *,
    model_label: str,
    model_path: Path,
    args: argparse.Namespace,
    eval_dir: Path,
) -> tuple[dict[str, float], Path | None]:
    """Evaluate a single model and persist its summary JSON."""

    # WHY: Persist plots in shared visualization tree so analysts keep a single source.
    shap_dir = SHAP_DIR_BY_MODEL.get(model_label, LIGHTGBM_SHAP_DIR)
    shap_seed = DEFAULT_RANDOM_STATE
    shap_sample_size: int | None = None
    shap_display = args.max_display
    if model_label == "random_forest":
        LOGGER.info(
            "Skipping SHAP computation for RandomForest to keep evaluation responsive.",
        )
        shap_display = 0

    try:
        raw_metrics = evaluate_lightgbm_on_test(
            parquet_path=Path(args.parquet),
            model_path=model_path,
            target_column=args.target_column,
            shap_output_dir=shap_dir,
            shap_max_display=shap_display,
            model_label=model_label,
            shap_sample_size=shap_sample_size,
            shap_random_state=shap_seed,
        )
    except Exception as exc:  # pragma: no cover - defensive: keep CLI resilient
        # WHY: Evaluation or downstream SHAP plotting for some model types may
        # raise errors (heavy deps, unexpected model objects). For RandomForest
        # we prefer to log the failure and continue the overall CLI flow rather
        # than raising and breaking the whole run.
        LOGGER.error(
            "Evaluation failed for model '%s' at path %s: %s",
            model_label,
            model_path,
            exc,
            exc_info=True,
        )
        return {}, None
    metrics = raw_metrics.copy()
    shap_payload = metrics.pop("shap", None)
    if isinstance(shap_payload, dict) and "plot_path" in shap_payload:
        shap_payload = {
            **shap_payload,
            "plot_path": str(to_project_relative_path(shap_payload["plot_path"])),
        }
    parquet_path = Path(args.parquet)
    summary = {
        "model_path": str(to_project_relative_path(model_path)),
        "parquet_path": str(to_project_relative_path(parquet_path)),
        "target_column": args.target_column,
        "metrics": metrics,
        "shap": shap_payload,
    }
    json_path = eval_dir / f"{model_label}_test_metrics.json"
    saved_json = save_training_results(summary, json_path)
    LOGGER.info("%s evaluation summary saved to %s", model_label.title(), saved_json)
    return metrics, saved_json


def evaluate_ridge_and_record(
    *,
    model_path: Path,
    scaler_path: Path,
    args: argparse.Namespace,
    eval_dir: Path,
) -> tuple[dict[str, float], Path | None]:
    """Evaluate Ridge and persist its summary JSON."""

    try:
        raw_metrics = evaluate_ridge_on_test(
            parquet_path=Path(args.parquet),
            model_path=model_path,
            scaler_path=scaler_path,
            target_column=args.target_column,
            shap_output_dir=None,
            shap_max_display=0,
            model_label="ridge",
        )
    except Exception as exc:  # pragma: no cover - defensive: keep CLI resilient
        LOGGER.error(
            "Evaluation failed for Ridge at path %s: %s",
            model_path,
            exc,
            exc_info=True,
        )
        return {}, None
    metrics = raw_metrics.copy()
    metrics.pop("shap", None)
    parquet_path = Path(args.parquet)
    summary = {
        "model_path": str(to_project_relative_path(model_path)),
        "parquet_path": str(to_project_relative_path(parquet_path)),
        "target_column": args.target_column,
        "metrics": metrics,
        "shap": None,
    }
    json_path = eval_dir / "ridge_test_metrics.json"
    saved_json = save_training_results(summary, json_path)
    LOGGER.info("Ridge evaluation summary saved to %s", saved_json)
    return metrics, saved_json


def execute_selected_models(args: argparse.Namespace, selected: str) -> int:
    """Evaluate requested models and emit CLI-friendly output."""

    eval_dir = Path(args.eval_dir)
    saved_lightgbm: Path | None = None

    if selected in {"lightgbm", "both", "all"}:
        LOGGER.info("Evaluating LightGBM on test split")
        metrics, saved_lightgbm = evaluate_model_and_record(
            model_label="lightgbm",
            model_path=Path(args.model),
            args=args,
            eval_dir=eval_dir,
        )
        metrics_json = json.dumps(metrics)
        LOGGER.info("LightGBM metrics: %s", metrics_json)
        sys.stdout.write(
            f"{datetime.now().isoformat()} | INFO | src.eval.main | LightGBM metrics: {metrics_json}\n"
        )
        sys.stdout.flush()
        LOGGER.info("LightGBM summary saved to %s", saved_lightgbm)
        sys.stdout.write(
            f"{datetime.now().isoformat()} | INFO | src.eval.main | LightGBM summary saved to {saved_lightgbm}\n"
        )
        sys.stdout.flush()

    rf_path = Path(args.rf_model)
    if selected in {"random_forest", "both", "all"}:
        if not rf_path.exists():
            if selected == "random_forest":
                LOGGER.error("RandomForest model not found; expected at %s", rf_path)
                return 1
            LOGGER.info("RandomForest model not found; skipping evaluation: %s", rf_path)
        else:
            LOGGER.info("Evaluating RandomForest on test split")
            rf_metrics, rf_summary = evaluate_model_and_record(
                model_label="random_forest",
                model_path=rf_path,
                args=args,
                eval_dir=eval_dir,
            )
            LOGGER.info("RandomForest metrics: %s", json.dumps(rf_metrics))
            LOGGER.info("RandomForest summary saved to %s", rf_summary)

    if selected in {"ridge", "all"} or (selected == "both" and False):
        ridge_model_path = Path(args.ridge_model)
        ridge_scaler_path = Path(args.ridge_scaler)
        if not ridge_model_path.exists():
            LOGGER.error("Ridge model not found; expected at %s", ridge_model_path)
            # Continue instead of exiting to not break multi-model runs
        else:
            LOGGER.info("Evaluating Ridge on test split")
            ridge_metrics, ridge_summary = evaluate_ridge_and_record(
                model_path=ridge_model_path,
                scaler_path=ridge_scaler_path,
                args=args,
                eval_dir=eval_dir,
            )
            LOGGER.info("Ridge metrics: %s", json.dumps(ridge_metrics))
            LOGGER.info("Ridge summary saved to %s", ridge_summary)

    if saved_lightgbm is not None:
        LOGGER.info("LightGBM evaluation artifact confirmed at %s", saved_lightgbm)
        sys.stdout.write(
            f"{datetime.now().isoformat()} | INFO | src.eval.main | LightGBM evaluation artifact confirmed at {saved_lightgbm}\n"
        )
        sys.stdout.flush()
    # No explicit stdout line for Ridge to keep logs concise; JSON is persisted.

    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation CLI workflow end-to-end."""

    LOGGER.debug("Starting evaluation CLI")

    parser = build_arg_parser()
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv_list)
    selected = determine_model_selection(args)
    return execute_selected_models(args, selected)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
