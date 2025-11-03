"""CLI helpers to evaluate trained models on the held-out test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.constants import (
    TARGET_COLUMN,
    DEFAULT_TRAINING_PARQUET_PATH,
    DEFAULT_LIGHTGBM_MODEL_FILE,
    DEFAULT_RANDOM_FOREST_MODEL_FILE,
    DEFAULT_EVAL_RESULTS_DIR,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_EVAL_N_JOBS,
    DEFAULT_EVAL_MAX_DISPLAY,
    DEFAULT_RANDOM_STATE,
)
from src.eval.eval import evaluate_lightgbm_on_test
from src.utils import (
    ensure_numeric_columns,
    load_splits_from_parquet,
    save_training_results,
    get_logger,
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
        choices=["lightgbm", "random_forest", "both"],
        default=None,
        help=(
            "Which model(s) to evaluate. When omitted, an interactive prompt "
            "selects between LightGBM, RandomForest, or both."
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
                "Select models to evaluate: [1] LightGBM, [2] RandomForest, [3] Both (default 3): "
            ).strip()
            if choice == "1":
                return "lightgbm"
            if choice == "2":
                return "random_forest"
    except Exception:
        LOGGER.debug("Interactive selection failed; defaulting to both models")

    return "both"


def evaluate_model_and_record(
    *,
    model_label: str,
    model_path: Path,
    args: argparse.Namespace,
    eval_dir: Path,
) -> tuple[dict[str, float], str]:
    """Evaluate a single model and persist its summary JSON."""

    # WHY: Persist plots in shared visualization tree so analysts keep a single source.
    shap_dir = project_root / "plots" / "shape" / ("LightGBM" if model_label == "lightgbm" else "rf")
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
            parquet_path=args.parquet,
            model_path=str(model_path),
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
        return {}, ""
    metrics = raw_metrics.copy()
    shap_payload = metrics.pop("shap", None)
    if isinstance(shap_payload, dict) and "plot_path" in shap_payload:
        shap_payload = {
            **shap_payload,
            "plot_path": str(to_project_relative_path(shap_payload["plot_path"])),
        }
    summary = {
        "model_path": str(to_project_relative_path(model_path)),
        "parquet_path": str(to_project_relative_path(args.parquet)),
        "target_column": args.target_column,
        "metrics": metrics,
        "shap": shap_payload,
    }
    json_path = eval_dir / f"{model_label}_test_metrics.json"
    saved_json = save_training_results(summary, str(json_path))
    LOGGER.info("%s evaluation summary saved to %s", model_label.title(), saved_json)
    return metrics, saved_json


def execute_selected_models(args: argparse.Namespace, selected: str) -> int:
    """Evaluate requested models and emit CLI-friendly output."""

    eval_dir = Path(args.eval_dir)
    saved_lightgbm: str | None = None

    if selected in {"lightgbm", "both"}:
        LOGGER.info("Evaluating LightGBM on test split")
        metrics, saved_lightgbm = evaluate_model_and_record(
            model_label="lightgbm",
            model_path=Path(args.model),
            args=args,
            eval_dir=eval_dir,
        )
        LOGGER.info("LightGBM metrics: %s", json.dumps(metrics))
        LOGGER.info("LightGBM summary saved to %s", saved_lightgbm)

    rf_path = Path(args.rf_model)
    if selected in {"random_forest", "both"}:
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

    if saved_lightgbm is not None:
        LOGGER.info("LightGBM evaluation artifact confirmed at %s", saved_lightgbm)

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
