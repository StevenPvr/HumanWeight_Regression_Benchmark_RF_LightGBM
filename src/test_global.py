"""Run project unit tests sequentially in a fixed stage order."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import sys
from pathlib import Path
from typing import Sequence

import pytest  # type: ignore

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
# WHY: ensure pytest discovers modules when run as a script
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(extra_args: Sequence[str] | None = None) -> int:
    """Execute each stage test suite sequentially using pytest."""
    test_files: list[Path] = [
        PROJECT_ROOT / "src" / "data_cleaning" / "test_data_cleaning.py",
        PROJECT_ROOT / "src" / "data_preparation" / "test_data_preparation.py",
        PROJECT_ROOT / "src" / "hyperparameters_optimization" / "test_optimization.py",
        PROJECT_ROOT / "src" / "feature_engineering" / "test_feature_engineering.py",
        PROJECT_ROOT / "src" / "training" / "test_training.py",
        PROJECT_ROOT / "src" / "eval" / "test_eval.py",
        PROJECT_ROOT / "src" / "baseline" / "test_ridge.py",
    ]
    # WHY: run suites one by one to retain deterministic logging and isolate failures
    args_prefix: list[str] = list(extra_args or [])
    for test_path in test_files:
        print(f"[test_global] Running {test_path.relative_to(PROJECT_ROOT)}")
        exit_code = pytest.main(args_prefix + [str(test_path)])
        if exit_code != 0:
            return exit_code
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
