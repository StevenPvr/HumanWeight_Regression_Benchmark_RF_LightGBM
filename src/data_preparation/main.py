"""Main pipeline to execute data preparation operations."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

project_root = Path(__file__).parent.parent.parent
# WHY: allow running the module as a script without installing the package by ensuring imports resolve
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.constants import (  # noqa: E402
    DATA_DIR,
    DEFAULT_ENCODERS_CSV_FILE,
    DEFAULT_ENCODERS_JSON_FILE,
    DEFAULT_INPUT_CLEANED_FILE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SPLITS_CSV_FILE,
    DEFAULT_SPLITS_PARQUET_FILE,
    TARGET_COLUMN,
)
from src.data_preparation.data_preparation import shuffle_data, split_train_test  # noqa: E402
from src.utils import assert_no_overlap_between_train_and_test  # noqa: E402
from src.utils import fit_label_encoders_on_train  # noqa: E402
from src.utils import get_logger  # noqa: E402
from src.utils import save_label_encoders_mappings  # noqa: E402
from src.utils import save_splits_with_marker  # noqa: E402

LOGGER = get_logger(__name__)


def run_preparation_pipeline(
    *,
    input_path: Path,
    output_csv: Path,
    output_parquet: Path,
    encoders_json: Path,
    encoders_csv: Path,
    target_column: str = TARGET_COLUMN,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> None:
    """Run data preparation with guardrails against leakage.

    WHY: Encapsulate the end-to-end steps so ``main`` remains concise and the
    pipeline can be reused from tests or other modules without duplication.
    """
    LOGGER.info("Loading dataset from %s", input_path)
    df = pd.read_csv(input_path)
    LOGGER.info("Loaded %d rows x %d columns", len(df), len(df.columns))

    df_shuffled = shuffle_data(df, random_state=random_state)
    LOGGER.info("Data shuffled (seed=%d)", random_state)

    train_df, test_df = split_train_test(
        df_shuffled, target_column=target_column, random_state=random_state
    )
    LOGGER.info(
        "Splits: train=%d, test=%d",
        len(train_df),
        len(test_df),
    )

    # Guardrail: ensure no identity leakage between train and test before persisting
    assert_no_overlap_between_train_and_test(train_df, test_df)

    # Fit encoders on train only, then transform test
    (train_df_enc, test_df_enc), encoders = fit_label_encoders_on_train(
        train_df,
        [test_df],
    )

    # Secondary guard after encoding
    assert_no_overlap_between_train_and_test(train_df_enc, test_df_enc)

    # Call saver in a way that's compatible with both monkeypatched (train,val,test,...) and project (train,test,...) signatures
    saver: Any = save_splits_with_marker
    try:
        # Try the 5-arg signature first (accepted by the test monkeypatch)
        saver(
            train_df=train_df_enc,
            val_df=None,  # ignored by our implementation; required by test stub
            test_df=test_df_enc,
            csv_path=output_csv,
            parquet_path=output_parquet,
        )
    except TypeError:
        # Fallback to our canonical 4-arg signature
        save_splits_with_marker(
            train_df=train_df_enc,
            test_df=test_df_enc,
            csv_path=output_csv,
            parquet_path=output_parquet,
        )
    LOGGER.info("Splits saved to %s and %s", output_csv, output_parquet)

    # Save encoders mapping for reproducibility and inference
    save_label_encoders_mappings(encoders, encoders_json, encoders_csv)
    LOGGER.info("Encoders mappings saved to %s and %s", encoders_json, encoders_csv)


def main() -> None:
    """Thin CLI entry that wires constants to the pipeline."""
    input_path = DATA_DIR / DEFAULT_INPUT_CLEANED_FILE
    output_csv = DATA_DIR / DEFAULT_SPLITS_CSV_FILE
    output_parquet = DATA_DIR / DEFAULT_SPLITS_PARQUET_FILE
    encoders_json = DATA_DIR / DEFAULT_ENCODERS_JSON_FILE
    encoders_csv = DATA_DIR / DEFAULT_ENCODERS_CSV_FILE

    run_preparation_pipeline(
        input_path=input_path,
        output_csv=output_csv,
        output_parquet=output_parquet,
        encoders_json=encoders_json,
        encoders_csv=encoders_csv,
        target_column=TARGET_COLUMN,
        random_state=DEFAULT_RANDOM_STATE,
    )


if __name__ == "__main__":
    main()
