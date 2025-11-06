# LLM Instructions

**Different rules apply depending on file type.**

---

## 1. Code Standards (ALL files)

- Apply KISS & DRY principles
- Write minimal, concise code
- Follow PEP 8 conventions
- Prioritize readability over performance (unless explicitly requested)
- Comment to explain WHY, not WHAT

---

## 2. Python Modules (.py) - Rules for `src/` files

### 2.1 Type Hints & Functions
- Use type hints for all parameters and return values
- Max 50 lines per function
- One task per function (Single Responsibility)
- Docstrings in English (Google/NumPy style)

### 2.2 Architecture
- Reusable functions → `src/utils.py`
- ALL constants → `src/constants.py` (paths, magic numbers, configs, defaults)
- Never hardcode constants - import from `constants.py`
- Create only requested function (no extras unless asked)
- Module structure: `__init__.py`, main file, `main.py` (CLI), `test_*.py`

### 2.3 Imports
- `from __future__ import annotations` at top
- Order: stdlib → third-party → local
- No wildcard imports
- Check `requirements.txt` before adding dependencies
- Key deps: pandas, numpy, scikit-learn, lightgbm, optuna, shap, matplotlib, seaborn

### 2.4 Logging & Errors
- Use `get_logger(__name__)` from `src/utils.py`
- Add meaningful log messages (INFO for progress)
- Handle errors with exceptions
- Validate input parameters

### 2.5 Code Quality
- Check & remove unused variables/imports
- Check undeclared variables
- Investigate why unused (may indicate bugs)
- Run `get_errors` after writing/editing functions
- Fix all errors before completion
- Run black/ruff after functions (install if missing via `requirements-dev.txt`)
  - Catches issues early vs accumulating debt
  - Optional per function, but mandatory before push/PR

### 2.6 Testing
- Create unit test immediately after each function
- Use mocked data (`monkeypatch` for deps)
- Never use real data in tests
- Test edge cases (empty, None, invalid)
- Naming: `test_<module_name>.py`
- Use pytest fixtures

---

## 3. Jupyter Notebooks (.ipynb) - Rules for `notebooks/`

### What Applies
- KISS, DRY, PEP 8, readability, comments
- Clear markdown between cells
- Logical flow: load → analyze → visualize → conclude

### What Does NOT Apply
- No test files (exploratory nature)
- No docstrings (use markdown cells)
- No function requirement (inline code OK)
- No 50-line limit
- Type hints optional

### Best Practices
- Descriptive markdown headers
- Comments for complex ops
- Focused cells (one step)
- Clear visualizations with titles/labels
- Document insights in markdown
- Runnable top-to-bottom
- Load from `data/`, save plots to `plots/`
- Use pandas, matplotlib, seaborn

---

## 4. Data & Files (ALL code)

### Paths
- Use `pathlib.Path` (not string concat)
- Reference `src/constants.py` paths
- Absolute paths from `PROJECT_ROOT`
- Ex: `DATA_DIR / "dataset.csv"` not `"data/dataset.csv"`

### Locations & Formats
- Input: `data/`
- Results: `results/`
- Plots: `plots/`
- Intermediate: `.parquet` (performance)
- Final/readable: `.csv`
- Config/metadata: `.json`
- Models: `results/models/` (joblib)
- Metrics: `results/eval/` (JSON)
- Include timestamps/version in experiment filenames

---

## 5. Machine Learning (ML modules)

### Reproducibility
- Use `DEFAULT_RANDOM_STATE` from `constants.py`
- Set seeds for numpy, pandas, sklearn, model libs
- Document splits (train/val/test)

### Model Dev
- Separate hyperparameter optimization from training
- Validate on holdout test set
- Save models + hyperparameters
- Use cross-validation when appropriate
- Project uses: LightGBM & Random Forest (regression)

### Feature Engineering
- Apply same transforms to train/val/test
- Save encoders/scalers (`data/encoders_mappings.json`)
- Never leak val/test info into training

---

## 6. Interaction

- Never assume methodology/requirements
- Ask for clarification when unclear
- Specify file type (`.py` or `.ipynb`)
- Follow appropriate rules per type

---
