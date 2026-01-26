# Myoai

Myoai contains two Python training scripts for tabular myopia prediction using:

- **Traditional ML** (scikit-learn + optional XGBoost/LightGBM): `src/train_traditional_ml.py`
- **Deep Learning** (PyTorch FT-Transformer): `src/run_ft_transformer.py`

Both scripts support **three tasks** (depending on which label columns exist in your CSV):

- **Myopia classification**: `label_myopia`
- **Fast progression classification**: `label_fast_progression`
- **Regression**: `label_regression` (or `regression_label` / `label_current_se`)

## What’s in this repo

- `src/train_traditional_ml.py`: traditional ML baseline + tuning + internal/external evaluation + bootstrap CIs.
- `src/run_ft_transformer.py`: FT-Transformer training + optional Optuna tuning + internal/external evaluation + bootstrap CIs.

> Note: This repository is intended to contain **code only** (no raw data, no personal information, no training outputs).

## Data requirements (CSV)

Your input is a single CSV file with one row per sample.
The scripts expect canonical column names (listed below). If your dataset uses different names, use `--column-map`.

### Required numeric feature columns

Both scripts require:

- One **age** column: `age` or `age_baseline`
- One **follow-up time** column: `days_since_first_test` (or `followup_days` / `days` / `time_days`)
- Baseline refraction measurements:
  - `baseline_sphere_self`
  - `baseline_cylinder_self`
  - `baseline_axis_self`
  - `baseline_sphere_other`
  - `baseline_cylinder_other`
  - `baseline_axis_other`

The scripts will coerce numeric columns to floats. For regression labels, very large absolute values (e.g., `> 50`) are interpreted as “stored in centi-diopters” and divided by 100.

### Optional categorical feature columns

If present, these columns are used as categorical inputs:

- `sex` (or `gender`)
- `intervention_method`
- `lens_function_left`
- `lens_function_right`

Categorical handling differs by script:

- Traditional ML: impute missing with the most frequent value, then one-hot encode (unknown categories in eval are ignored).
- FT-Transformer: fill missing with `unknown`, map categories to integer IDs, then embed them (unknown categories in eval map to ID 0).

### Label columns (tasks)

If label columns are missing, the corresponding task will be skipped or will error (depending on the script and context).

- Classification labels should be one of: `yes/no`, `true/false`, `1/0`.
- Regression label should be numeric (or convertible to numeric).

## Column mapping

If your CSV uses different column names, provide a JSON mapping file and pass `--column-map`.

Example (`column_map.json`):

```json
{
  "my_source_age_col": "age",
  "my_source_days_col": "days_since_first_test",
  "my_source_label_col": "label_myopia"
}
```

## Installation

Recommended: Python 3.10+.

Minimal packages:

- `numpy`, `pandas`, `scikit-learn`
- For FT-Transformer: `torch`
- Optional (for tuning): `optuna`
- Optional (for some traditional models): `xgboost`, `lightgbm`

## Quick start

### Traditional ML (3 tasks, internal+external evaluation)

Run with internal train split + evaluate on an external CSV:

```bash
python src/train_traditional_ml.py \
  --train-data path/to/internal.csv \
  --eval-data path/to/external.csv \
  --run-name my_run \
  --tasks all
```

### FT-Transformer (3 tasks, internal+external evaluation)

```bash
python src/run_ft_transformer.py \
  --data path/to/internal.csv \
  --eval-data path/to/external.csv \
  --tasks all
```

## Outputs

Both scripts write metrics and (optionally) models/predictions under output directories controlled by CLI flags.
This repo intentionally does **not** track those outputs in git.

## Privacy & publishing

Do not commit any raw CSV/Excel files, especially if they contain personal identifiers (phone numbers, addresses, etc.).
