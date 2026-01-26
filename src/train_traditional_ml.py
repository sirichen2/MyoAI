from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor

    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

try:
    import optuna
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except Exception:
    optuna = None  # type: ignore[assignment]
    TPESampler = None  # type: ignore[assignment]
    OPTUNA_AVAILABLE = False


UNKNOWN_CATEGORY = "unknown"
STANDARD_DAYS_COLUMN = "days_since_first_test"

CLASS_TASKS: Dict[str, str] = {
    "myopia": "label_myopia",
    "fast_progress": "label_fast_progression",
}

REGRESSION_LABEL_CANDIDATES = [
    "label_regression",
    "regression_label",
    "label_current_se",
]

YES_TOKENS = {"yes", "y", "true", "1"}
NO_TOKENS = {"no", "n", "false", "0"}


def resolve_first_present_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def extract_binary_labels(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    normalized = series.astype(str).str.strip().str.lower()
    mask = normalized.isin(YES_TOKENS | NO_TOKENS).to_numpy()
    y = (normalized[mask].isin(YES_TOKENS)).astype(int).to_numpy()
    return mask, y


def load_column_map(path: Optional[Path]) -> Optional[Dict[str, str]]:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--column-map must be a JSON object mapping old->new column names.")
    result: Dict[str, str] = {}
    for k, v in payload.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError("--column-map values must be strings.")
        result[k] = v
    return result


def apply_column_map(df: pd.DataFrame, column_map: Optional[Dict[str, str]]) -> pd.DataFrame:
    if not column_map:
        return df
    renamed = df.rename(columns=column_map)
    targets = list(column_map.values())
    duplicates = sorted({t for t in targets if targets.count(t) > 1})
    if duplicates:
        raise ValueError(f"--column-map produces duplicate target columns: {duplicates}")
    return renamed


def to_diopters(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mask = values.abs() > 50
    if mask.any():
        values.loc[mask] = values.loc[mask] / 100.0
    return values


def resolve_regression_label(df_train: pd.DataFrame, df_eval: pd.DataFrame) -> Optional[str]:
    for col in REGRESSION_LABEL_CANDIDATES:
        if col in df_train.columns and col in df_eval.columns:
            return col
    return None

BASELINE_FEATURES = [
    "baseline_sphere_self",
    "baseline_cylinder_self",
    "baseline_axis_self",
    "baseline_sphere_other",
    "baseline_cylinder_other",
    "baseline_axis_other",
]

CAT_FEATURE_GROUPS: List[List[str]] = [
    ["sex", "gender"],
    ["intervention_method"],
    ["lens_function_left"],
    ["lens_function_right"],
]


def parse_days_to_float(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    match = re.search(r"(-?\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else float("nan")


def select_age_column(df: pd.DataFrame) -> str:
    for candidate in ["age_baseline", "age"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("Missing required age column: expected `age` or `age_baseline`.")


def build_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    age_col = select_age_column(df)
    distance_col = None
    for candidate in [STANDARD_DAYS_COLUMN, "followup_days", "days", "time_days"]:
        if candidate in df.columns:
            distance_col = candidate
            break
    if distance_col is None:
        raise ValueError(
            "Missing required follow-up days column: expected `days_since_first_test` "
            "(or `followup_days`/`days`/`time_days`)."
        )

    df = df.copy()
    df[STANDARD_DAYS_COLUMN] = df[distance_col].map(parse_days_to_float)
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")

    for col in BASELINE_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing required baseline feature: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cat_cols: List[str] = []
    for candidates in CAT_FEATURE_GROUPS:
        col = resolve_first_present_column(df, candidates)
        if col:
            cat_cols.append(col)
    feature_cols = [age_col, STANDARD_DAYS_COLUMN] + BASELINE_FEATURES + cat_cols
    X = df[feature_cols].copy()
    numeric_cols = [c for c in feature_cols if c not in cat_cols]
    return X, numeric_cols, cat_cols


def build_preprocessor(numeric_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def metric_sens_spec(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return float(sens), float(spec)


def bootstrap_ci(
    rng: np.random.Generator,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    metric_fn,
    n_boot: int,
) -> Tuple[float, float]:
    values = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        yp_prob = y_prob[idx] if y_prob is not None else None
        try:
            v = metric_fn(yt, yp, yp_prob)
        except Exception:
            continue
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        values.append(float(v))
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def bootstrap_classification_cis(
    rng: np.random.Generator,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int,
) -> Dict[str, Tuple[float, float]]:
    if n_boot <= 0:
        return {}
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = int(len(y_true))
    if n <= 1:
        return {
            "auc": (float("nan"), float("nan")),
            "acc": (float("nan"), float("nan")),
            "f1": (float("nan"), float("nan")),
            "sens": (float("nan"), float("nan")),
            "spec": (float("nan"), float("nan")),
        }

    aucs = np.full(int(n_boot), np.nan, dtype=float)
    accs = np.full(int(n_boot), np.nan, dtype=float)
    f1s = np.full(int(n_boot), np.nan, dtype=float)
    senss = np.full(int(n_boot), np.nan, dtype=float)
    specs = np.full(int(n_boot), np.nan, dtype=float)

    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        yhat = (yp >= 0.5).astype(int)

        if np.unique(yt).size > 1:
            try:
                aucs[i] = float(roc_auc_score(yt, yp))
            except Exception:
                aucs[i] = np.nan
        accs[i] = float(np.mean(yhat == yt))

        tp = int(np.sum((yt == 1) & (yhat == 1)))
        fp = int(np.sum((yt == 0) & (yhat == 1)))
        fn = int(np.sum((yt == 1) & (yhat == 0)))
        tn = int(np.sum((yt == 0) & (yhat == 0)))

        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        senss[i] = float(sens)
        specs[i] = float(spec)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = sens if not math.isnan(sens) else 0.0
        f1s[i] = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    def ci(values: np.ndarray) -> Tuple[float, float]:
        if np.all(np.isnan(values)):
            return float("nan"), float("nan")
        return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))

    return {
        "auc": ci(aucs),
        "acc": ci(accs),
        "f1": ci(f1s),
        "sens": ci(senss),
        "spec": ci(specs),
    }


def bootstrap_regression_cis(
    rng: np.random.Generator,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int,
) -> Dict[str, Tuple[float, float]]:
    if n_boot <= 0:
        return {}
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = int(len(y_true))
    if n <= 1:
        return {
            "r2": (float("nan"), float("nan")),
            "mae": (float("nan"), float("nan")),
            "rmse": (float("nan"), float("nan")),
        }

    r2s = np.full(int(n_boot), np.nan, dtype=float)
    maes = np.full(int(n_boot), np.nan, dtype=float)
    rmses = np.full(int(n_boot), np.nan, dtype=float)

    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        err = yt - yp
        maes[i] = float(np.mean(np.abs(err)))
        rmses[i] = float(np.sqrt(np.mean(err**2)))
        denom = float(np.sum((yt - float(np.mean(yt))) ** 2))
        if denom > 0:
            r2s[i] = float(1.0 - float(np.sum(err**2)) / denom)

    def ci(values: np.ndarray) -> Tuple[float, float]:
        if np.all(np.isnan(values)):
            return float("nan"), float("nan")
        return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))

    return {
        "r2": ci(r2s),
        "mae": ci(maes),
        "rmse": ci(rmses),
    }

def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["auc"] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    metrics["acc"] = accuracy_score(y_true, y_pred)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    sens, spec = metric_sens_spec(y_true, y_pred)
    metrics["sens"] = sens
    metrics["spec"] = spec
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["r2"] = r2_score(y_true, y_pred)
    metrics["mae"] = mean_absolute_error(y_true, y_pred)
    metrics["rmse"] = math.sqrt(mean_squared_error(y_true, y_pred))
    return metrics


def get_class_models(seed: int) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=500, random_state=seed, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=3000, random_state=seed),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
        "Decision Tree": DecisionTreeClassifier(random_state=seed, max_depth=15),
        "Naive Bayes": GaussianNB(),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=600,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=4,
        )
    if LGB_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            objective="binary",
            random_state=seed,
            n_estimators=800,
            learning_rate=0.05,
        )
    # keep stable order
    preferred = [
        "XGBoost",
        "Gradient Boosting",
        "Random Forest",
        "LightGBM",
        "Extra Trees",
        "Logistic Regression",
        "K-Nearest Neighbors",
        "Decision Tree",
        "Naive Bayes",
    ]
    return {k: models[k] for k in preferred if k in models}


def get_reg_models(seed: int) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "Gradient Boosting": GradientBoostingRegressor(random_state=seed),
        "Random Forest": RandomForestRegressor(n_estimators=600, random_state=seed, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=800, random_state=seed, n_jobs=-1),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=15),
        "Decision Tree": DecisionTreeRegressor(random_state=seed, max_depth=15),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=800,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=4,
        )
    if LGB_AVAILABLE:
        models["LightGBM"] = LGBMRegressor(
            objective="regression",
            random_state=seed,
            n_estimators=800,
            learning_rate=0.05,
        )
    preferred = [
        "XGBoost",
        "Gradient Boosting",
        "Random Forest",
        "LightGBM",
        "Extra Trees",
        "K-Nearest Neighbors",
        "Decision Tree",
    ]
    return {k: models[k] for k in preferred if k in models}


def normalize_model_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(name)).strip("_").lower()
    return slug or "model"


def parse_tune_models(value: str) -> Optional[Set[str]]:
    raw = str(value or "").strip()
    if not raw or raw.lower() == "all":
        return None
    if raw.lower() in {"none", "off", "false", "0"}:
        return set()
    return {normalize_model_name(part) for part in raw.split(",") if part.strip()}


def parse_tasks(value: str) -> Optional[Set[str]]:
    raw = str(value or "").strip()
    if not raw or raw.lower() == "all":
        return None
    if raw.lower() in {"none", "off", "false", "0"}:
        return set()
    allowed = set(CLASS_TASKS.keys()) | {"regression"}
    tasks = {part.strip() for part in raw.split(",") if part.strip()}
    unknown = sorted(t for t in tasks if t not in allowed)
    if unknown:
        raise ValueError(f"Unknown --tasks: {unknown}. Allowed: {sorted(allowed)}")
    return tasks


def load_best_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def normalize_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan", "-"}:
        return None
    return value


def load_best_params_from_dir(best_dir: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Load best hyperparameters from tuning *.best.json files.
    Returns mapping: (task_key, model_name) -> best_params dict.
    """
    mapping: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not best_dir.exists():
        return mapping
    for path in best_dir.glob("*.best.json"):
        payload = load_best_json(path)
        if not payload:
            continue
        if payload.get("skipped") is True:
            continue
        task = payload.get("task")
        model = payload.get("model")
        best_params = payload.get("best_params")
        if not isinstance(task, str) or not isinstance(model, str) or not isinstance(best_params, dict):
            continue
        cleaned = {k: normalize_none(v) for k, v in best_params.items()}
        mapping[(task, model)] = cleaned
    return mapping


def parse_run_models(value: str) -> Optional[Set[str]]:
    raw = str(value or "").strip()
    if not raw or raw.lower() == "all":
        return None
    if raw.lower() in {"none", "off", "false", "0"}:
        return set()
    return {normalize_model_name(part) for part in raw.split(",") if part.strip()}


def model_has_search_space(model_name: str) -> bool:
    return normalize_model_name(model_name) in {
        "gradientboosting",
        "randomforest",
        "extratrees",
        "logisticregression",
        "knearestneighbors",
        "decisiontree",
        "naivebayes",
        "xgboost",
        "lightgbm",
    }


def is_xgboost_model(model_name: str) -> bool:
    return normalize_model_name(model_name) == "xgboost"


def is_lightgbm_model(model_name: str) -> bool:
    return normalize_model_name(model_name) == "lightgbm"


def safe_cv_splits_classification(y: np.ndarray, requested: int) -> int:
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    return max(0, min(int(requested), pos, neg))


def safe_cv_splits_regression(y: np.ndarray, requested: int) -> int:
    return max(0, min(int(requested), int(len(y))))


def suggest_params_for_model(trial: Any, model_name: str, task_type: str) -> Dict[str, Any]:
    key = normalize_model_name(model_name)
    is_classification = task_type == "classification"

    if key == "knearestneighbors":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 51, step=2),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_categorical("p", [1, 2]),
        }

    if key == "logisticregression":
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
        return {
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "solver": solver,
            "penalty": "l2",
        }

    if key == "naivebayes":
        return {
            "var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-7, log=True),
        }

    if key == "decisiontree":
        max_depth_choices: List[Optional[int]] = [None] + list(range(2, 31))
        criterion_choices = ["gini", "entropy", "log_loss"] if is_classification else ["squared_error", "friedman_mse"]
        return {
            "criterion": trial.suggest_categorical("criterion", criterion_choices),
            "max_depth": trial.suggest_categorical("max_depth", max_depth_choices),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        }

    if key in {"randomforest", "extratrees"}:
        max_depth_choices = [None] + list(range(3, 31))
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=50),
            "max_depth": trial.suggest_categorical("max_depth", max_depth_choices),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        }

    if key == "gradientboosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        }

    if key == "xgboost":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        }

    if key == "lightgbm":
        max_depth = trial.suggest_int("max_depth", -1, 12)
        num_leaves = trial.suggest_int("num_leaves", 15, 255)
        if max_depth > 0:
            max_leaves = 2 ** max_depth
            num_leaves = min(num_leaves, max_leaves)
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        }

    return {}


def tune_model_optuna(
    *,
    base_model: Any,
    model_name: str,
    task_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    preprocessor: ColumnTransformer,
    scoring: str,
    n_trials: int,
    cv: int,
    seed: int,
    n_jobs: int,
    timeout: Optional[float],
    early_stopping_rounds: int,
    max_estimators: int,
) -> Tuple[Dict[str, Any], float, Any]:
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna is required for --tune. Install with: pip install optuna")

    if task_type == "classification":
        cv_splits = safe_cv_splits_classification(y_train, cv)
        if cv_splits < 2:
            raise RuntimeError(
                "Not enough samples per class for StratifiedKFold: "
                f"cv={cv}, pos={int(np.sum(y_train==1))}, neg={int(np.sum(y_train==0))}"
            )
        splitter = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    elif task_type == "regression":
        cv_splits = safe_cv_splits_regression(y_train, cv)
        if cv_splits < 2:
            raise RuntimeError(f"Not enough samples for KFold: cv={cv}, n={len(y_train)}")
        splitter = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    use_early_stopping = early_stopping_rounds > 0 and (
        is_xgboost_model(model_name) or is_lightgbm_model(model_name)
    )

    sampler = TPESampler(seed=seed)  # type: ignore[misc]
    study = optuna.create_study(direction="maximize", sampler=sampler)  # type: ignore[union-attr]

    def objective(trial: Any) -> float:
        params = suggest_params_for_model(trial, model_name, task_type)
        if not params:
            return -1e9

        if is_xgboost_model(model_name):
            if use_early_stopping:
                params["n_estimators"] = int(max_estimators)
            else:
                params["n_estimators"] = trial.suggest_int("n_estimators", 200, 1200, step=50)
            params["eval_metric"] = "auc" if task_type == "classification" else "rmse"

        if is_lightgbm_model(model_name):
            if use_early_stopping:
                params["n_estimators"] = int(max_estimators)
            else:
                params["n_estimators"] = trial.suggest_int("n_estimators", 200, 2000, step=50)
            params["verbosity"] = -1

        if task_type == "classification" and (is_xgboost_model(model_name) or is_lightgbm_model(model_name)):
            pos = int(np.sum(y_train == 1))
            neg = int(np.sum(y_train == 0))
            ratio = (neg / pos) if pos else 1.0
            low = max(0.1, ratio / 5.0)
            high = min(100.0, ratio * 5.0)
            if high <= low:
                high = low * 1.01
            params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", low, high, log=True)

        model = clone(base_model)
        try:
            model.set_params(**params)
        except Exception:
            return -1e9

        if use_early_stopping:
            scores: List[float] = []
            best_rounds: List[float] = []
            split_iter = splitter.split(X_train, y_train if task_type == "classification" else None)
            for fold_train_idx, fold_val_idx in split_iter:
                X_tr = X_train.iloc[fold_train_idx]
                y_tr = y_train[fold_train_idx]
                X_val = X_train.iloc[fold_val_idx]
                y_val = y_train[fold_val_idx]

                prep = clone(preprocessor)
                X_tr_t = prep.fit_transform(X_tr, y_tr)
                X_val_t = prep.transform(X_val)

                fold_model = clone(model)
                try:
                    if is_xgboost_model(model_name):
                        fit_kwargs: Dict[str, Any] = {"eval_set": [(X_val_t, y_val)], "verbose": False}
                        if early_stopping_rounds > 0:
                            fold_model.set_params(early_stopping_rounds=int(early_stopping_rounds))
                        fold_model.fit(X_tr_t, y_tr, **fit_kwargs)
                        best_iter = getattr(fold_model, "best_iteration", None)
                        if best_iter is None:
                            best_iter = getattr(fold_model, "best_iteration_", None)
                        best_n = int(best_iter) + 1 if best_iter is not None else int(params["n_estimators"])

                        pred_kwargs: Dict[str, Any] = {}
                        if best_iter is not None:
                            pred_kwargs["iteration_range"] = (0, int(best_iter) + 1)
                        if task_type == "classification":
                            try:
                                y_prob = fold_model.predict_proba(X_val_t, **pred_kwargs)[:, 1]
                            except TypeError:
                                y_prob = fold_model.predict_proba(X_val_t)[:, 1]
                            score = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else np.nan
                        else:
                            try:
                                y_pred = fold_model.predict(X_val_t, **pred_kwargs)
                            except TypeError:
                                y_pred = fold_model.predict(X_val_t)
                            score = r2_score(y_val, y_pred)
                    else:
                        import lightgbm as lgb  # type: ignore[import-not-found]

                        callbacks = []
                        if early_stopping_rounds > 0:
                            callbacks.append(
                                lgb.early_stopping(
                                    stopping_rounds=int(early_stopping_rounds),
                                    first_metric_only=True,
                                    verbose=False,
                                )
                            )
                        fit_kwargs = {"eval_set": [(X_val_t, y_val)], "callbacks": callbacks}
                        fit_kwargs["eval_metric"] = "auc" if task_type == "classification" else "l2"
                        fold_model.fit(X_tr_t, y_tr, **fit_kwargs)

                        best_iter = getattr(fold_model, "best_iteration_", None)
                        best_n = int(best_iter) if best_iter else int(params["n_estimators"])

                        pred_kwargs = {}
                        if best_iter:
                            pred_kwargs["num_iteration"] = int(best_iter)
                        if task_type == "classification":
                            y_prob = fold_model.predict_proba(X_val_t, **pred_kwargs)[:, 1]
                            score = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else np.nan
                        else:
                            y_pred = fold_model.predict(X_val_t, **pred_kwargs)
                            score = r2_score(y_val, y_pred)
                except Exception:
                    return -1e9

                scores.append(float(score))
                best_rounds.append(float(best_n))

            score = float(np.nanmean(np.array(scores, dtype=float)))
            if math.isnan(score):
                return -1e9
            if best_rounds:
                trial.set_user_attr("best_n_estimators", int(np.nanmedian(np.array(best_rounds, dtype=float))))
            return score

        pipe = Pipeline([("prep", clone(preprocessor)), ("model", model)])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            scores = cross_val_score(
                pipe,
                X_train,
                y_train,
                cv=splitter,
                scoring=scoring,
                n_jobs=n_jobs,
                error_score=np.nan,
            )
        score = float(np.nanmean(scores))
        if math.isnan(score):
            return -1e9
        return score

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)  # type: ignore[union-attr]
    best_params = dict(study.best_params)
    if use_early_stopping:
        best_n = study.best_trial.user_attrs.get("best_n_estimators")  # type: ignore[union-attr]
        if isinstance(best_n, int) and best_n > 0:
            best_params["n_estimators"] = int(best_n)
    return best_params, float(study.best_value), study


def ensure_dirs(run_dir: Path) -> Dict[str, Path]:
    paths = {
        "run": run_dir,
        "models": run_dir / "models",
        "metrics": run_dir / "metrics",
        "predictions": run_dir / "predictions",
        "logs": run_dir / "logs",
        "tuning": run_dir / "tuning",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train traditional ML models with Int/Ext eval + 95% CI.")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("outputs"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated tasks to run: myopia,fast_progress,regression (default: all).",
    )
    parser.add_argument(
        "--column-map",
        type=Path,
        default=None,
        help="Optional JSON file mapping input column names to canonical names expected by this script.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model names to run (default: all). Example: randomforest,extratrees,gradientboosting",
    )
    parser.add_argument(
        "--load-best-dir",
        type=Path,
        default=None,
        help="Directory containing tuning *.best.json files; loads best_params for final training (works without --tune).",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Use Optuna Bayesian optimization to tune hyperparameters with CV on the internal training split.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If tuning best.json already exists in the run output, reuse it and only tune missing models/tasks.",
    )
    parser.add_argument("--tune-trials", type=int, default=50)
    parser.add_argument("--tune-cv", type=int, default=5)
    parser.add_argument(
        "--tune-models",
        type=str,
        default="all",
        help="Comma-separated model names to tune (default: all). Example: xgboost,lightgbm,randomforest",
    )
    parser.add_argument("--tune-jobs", type=int, default=1, help="Parallelism for CV scoring in tuning.")
    parser.add_argument("--tune-timeout", type=float, default=None, help="Optional timeout (seconds) per study.")
    parser.add_argument(
        "--tune-early-stopping-rounds",
        type=int,
        default=50,
        help="Early stopping rounds for XGBoost/LightGBM during tuning (0 disables early stopping).",
    )
    parser.add_argument(
        "--tune-max-estimators",
        type=int,
        default=2000,
        help="Max boosting rounds for XGBoost/LightGBM when early stopping is enabled.",
    )
    args = parser.parse_args()

    start_time = datetime.now().isoformat()
    rng = np.random.default_rng(args.seed)
    tune_models = parse_tune_models(args.tune_models)
    tasks = parse_tasks(args.tasks)
    run_models = parse_run_models(args.models)
    allowed_tasks = set(CLASS_TASKS.keys()) | {"regression"}
    if tasks is None or tasks == allowed_tasks:
        tasks_tag = "all"
    elif not tasks:
        tasks_tag = "none"
    else:
        tasks_tag = "_".join(sorted(tasks))
    tuning_records: List[Dict[str, Any]] = []
    if args.tune and not OPTUNA_AVAILABLE and tune_models != set():
        raise RuntimeError("Optuna is required for --tune. Install with: pip install optuna")

    run_dir = args.out_root / args.run_name
    paths = ensure_dirs(run_dir)

    column_map = load_column_map(args.column_map)
    df_train = apply_column_map(pd.read_csv(args.train_data), column_map)
    df_eval = apply_column_map(pd.read_csv(args.eval_data), column_map)
    resolved_class_tasks: Dict[str, str] = dict(CLASS_TASKS)
    df_eval = df_eval.copy()
    reg_label = resolve_regression_label(df_train, df_eval)

    # Build X and preprocessing for train and eval
    X_all, numeric_cols, cat_cols = build_feature_frame(df_train)
    preprocessor = build_preprocessor(numeric_cols, cat_cols)

    # Ensure eval has missing cat cols filled with placeholder
    for c in cat_cols:
        if c not in df_eval.columns:
            # If eval uses an alternative alias, copy it; otherwise fill with a safe placeholder.
            alias = None
            for group in CAT_FEATURE_GROUPS:
                if c in group:
                    alias = resolve_first_present_column(df_eval, [a for a in group if a != c])
                    break
            df_eval[c] = df_eval[alias] if alias else UNKNOWN_CATEGORY
    X_eval, _, _ = build_feature_frame(df_eval)

    config = {
        "train_data": str(args.train_data),
        "eval_data": str(args.eval_data),
        "test_size": args.test_size,
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
        "selected_tasks": args.tasks,
        "selected_models": args.models,
        "features": {
            "numeric": numeric_cols,
            "categorical": cat_cols,
        },
        "tasks": {
            "classification": resolved_class_tasks,
            "regression": reg_label,
        },
        "tuning": {
            "enabled": bool(args.tune),
            "resume": bool(args.resume),
            "load_best_dir": str(args.load_best_dir) if args.load_best_dir else None,
            "backend": "optuna",
            "n_trials": args.tune_trials,
            "cv": args.tune_cv,
            "n_jobs": args.tune_jobs,
            "timeout": args.tune_timeout,
            "early_stopping_rounds": args.tune_early_stopping_rounds,
            "max_estimators": args.tune_max_estimators,
            "scoring": {"classification": "roc_auc", "regression": "r2"},
            "models": args.tune_models,
        },
        "started_at": start_time,
    }
    save_json(paths["run"] / "run_config.json", config)

    metrics_rows: List[Dict[str, Any]] = []
    best_params_by_task_model: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if args.load_best_dir:
        best_params_by_task_model = load_best_params_from_dir(args.load_best_dir)

    # Split once and reuse across tasks for fairness
    indices = np.arange(len(df_train))
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=args.seed, shuffle=True
    )

    X_train = X_all.iloc[train_idx]
    X_test = X_all.iloc[test_idx]

    # === Classification tasks ===
    class_models = get_class_models(args.seed)
    tuned_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for task_key, label_col in resolved_class_tasks.items():
        if tasks is not None and task_key not in tasks:
            continue
        if label_col not in df_train.columns or label_col not in df_eval.columns:
            continue
        mask_train, y_train_masked = extract_binary_labels(df_train[label_col])
        y_train_full = np.zeros(len(df_train), dtype=int)
        y_train_full[mask_train] = y_train_masked

        mask_eval, y_eval_masked = extract_binary_labels(df_eval[label_col])
        y_eval_full = np.zeros(len(df_eval), dtype=int)
        y_eval_full[mask_eval] = y_eval_masked
        # apply split masks
        mask_train_split = mask_train[train_idx]
        mask_test_split = mask_train[test_idx]

        y_train = y_train_full[train_idx][mask_train_split]
        y_test = y_train_full[test_idx][mask_test_split]

        X_train_task = X_train.iloc[mask_train_split]
        X_test_task = X_test.iloc[mask_test_split]

        y_eval = y_eval_full[mask_eval]
        X_eval_task = X_eval.iloc[mask_eval]
        for model_name, model in class_models.items():
            if run_models is not None and normalize_model_name(model_name) not in run_models:
                continue
            model_key = normalize_model_name(model_name)
            tune_this = (
                bool(args.tune)
                and model_has_search_space(model_name)
                and (tune_models is None or model_key in tune_models)
            )
            tuned_params: Optional[Dict[str, Any]] = None
            loaded_params = best_params_by_task_model.get((task_key, model_name)) if best_params_by_task_model else None
            if loaded_params:
                tuned_params = loaded_params
            if tune_this:
                cache_key = (task_key, model_name)
                if cache_key not in tuned_cache:
                    best_json_path = paths["tuning"] / f"{task_key}__{slugify(model_name)}.best.json"
                    if args.resume:
                        existing = load_best_json(best_json_path)
                        existing_params = (existing or {}).get("best_params")
                        if isinstance(existing, dict) and existing.get("skipped") is True:
                            tuned_cache[cache_key] = {}
                            tuning_records.append(existing)
                        elif isinstance(existing_params, dict) and existing_params:
                            tuned_cache[cache_key] = existing_params
                            tuning_records.append(existing)  # type: ignore[arg-type]

                    if cache_key not in tuned_cache:
                        cv_used = safe_cv_splits_classification(y_train, args.tune_cv)
                        pos = int(np.sum(y_train == 1))
                        neg = int(np.sum(y_train == 0))
                        es_rounds = (
                            int(args.tune_early_stopping_rounds)
                            if (is_xgboost_model(model_name) or is_lightgbm_model(model_name))
                            else 0
                        )
                        max_est = int(args.tune_max_estimators) if es_rounds > 0 else 0

                        if cv_used < 2:
                            tuned_cache[cache_key] = {}
                            record = {
                                "task": task_key,
                                "task_type": "classification",
                                "model": model_name,
                                "scoring": "roc_auc",
                                "cv": cv_used,
                                "n_trials": args.tune_trials,
                                "pos": pos,
                                "neg": neg,
                                "neg_pos_ratio": (neg / pos) if pos else None,
                                "skipped": True,
                                "reason": f"Not enough samples per class for CV (pos={pos}, neg={neg}).",
                            }
                            tuning_records.append(record)
                            save_json(best_json_path, record)
                        else:
                            try:
                                tuned_params, tuned_cv, study = tune_model_optuna(
                                    base_model=model,
                                    model_name=model_name,
                                    task_type="classification",
                                    X_train=X_train_task,
                                    y_train=y_train,
                                    preprocessor=preprocessor,
                                    scoring="roc_auc",
                                    n_trials=args.tune_trials,
                                    cv=cv_used,
                                    seed=args.seed,
                                    n_jobs=args.tune_jobs,
                                    timeout=args.tune_timeout,
                                    early_stopping_rounds=es_rounds,
                                    max_estimators=max_est,
                                )
                                tuned_cache[cache_key] = tuned_params

                                record = {
                                    "task": task_key,
                                    "task_type": "classification",
                                    "model": model_name,
                                    "scoring": "roc_auc",
                                    "cv": cv_used,
                                    "n_trials": args.tune_trials,
                                    "pos": pos,
                                    "neg": neg,
                                    "neg_pos_ratio": (neg / pos) if pos else None,
                                    "early_stopping_rounds": es_rounds,
                                    "max_estimators": max_est,
                                    "best_value": tuned_cv,
                                    "best_params": tuned_params,
                                }
                                tuning_records.append(record)
                                save_json(best_json_path, record)
                                try:
                                    study_df = study.trials_dataframe()  # type: ignore[attr-defined]
                                    study_df.to_csv(
                                        paths["tuning"] / f"{task_key}__{slugify(model_name)}.trials.csv",
                                        index=False,
                                    )
                                except Exception:
                                    pass
                            except Exception as e:
                                tuned_cache[cache_key] = {}
                                record = {
                                    "task": task_key,
                                    "task_type": "classification",
                                    "model": model_name,
                                    "scoring": "roc_auc",
                                    "cv": cv_used,
                                    "n_trials": args.tune_trials,
                                    "pos": pos,
                                    "neg": neg,
                                    "neg_pos_ratio": (neg / pos) if pos else None,
                                    "early_stopping_rounds": es_rounds,
                                    "max_estimators": max_est,
                                    "skipped": True,
                                    "reason": str(e),
                                }
                                tuning_records.append(record)
                                save_json(best_json_path, record)
                tuned_params = tuned_cache.get(cache_key) or tuned_params

            tuned_model = clone(model)
            if tuned_params:
                tuned_model.set_params(**tuned_params)
            pipe = Pipeline([("prep", clone(preprocessor)), ("model", tuned_model)])
            pipe.fit(X_train_task, y_train)

            # internal
            prob_int = pipe.predict_proba(X_test_task)[:, 1]
            pred_int = (prob_int >= 0.5).astype(int)
            int_m = classification_metrics(y_test, pred_int, prob_int)

            # external
            prob_ext = pipe.predict_proba(X_eval_task)[:, 1]
            pred_ext = (prob_ext >= 0.5).astype(int)
            ext_m = classification_metrics(y_eval, pred_ext, prob_ext)

            ci = {}
            if args.n_bootstrap > 0:
                ci_int = bootstrap_classification_cis(rng, y_test, prob_int, args.n_bootstrap)
                ci_ext = bootstrap_classification_cis(rng, y_eval, prob_ext, args.n_bootstrap)
                for key in ["auc", "acc", "f1", "sens", "spec"]:
                    ci[f"{key}_int_lower"], ci[f"{key}_int_upper"] = ci_int.get(key, (math.nan, math.nan))
                    ci[f"{key}_ext_lower"], ci[f"{key}_ext_upper"] = ci_ext.get(key, (math.nan, math.nan))

            # save model
            model_path = paths["models"] / f"{task_key}__{model_name}.pkl"
            joblib.dump(pipe, model_path)

            # save predictions
            pred_dir = paths["predictions"] / task_key
            pred_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {"y_true": y_test, "y_prob": prob_int, "y_pred": pred_int}
            ).to_csv(pred_dir / f"{model_name}__internal.csv", index=False)
            pd.DataFrame(
                {"y_true": y_eval, "y_prob": prob_ext, "y_pred": pred_ext}
            ).to_csv(pred_dir / f"{model_name}__external.csv", index=False)

            metrics_rows.append(
                {
                    "Model": model_name,
                    "Task": task_key,
                    "Split": "both",
                    **{f"{k}_int": v for k, v in int_m.items()},
                    **{f"{k}_ext": v for k, v in ext_m.items()},
                    **ci,
                }
            )

    # === Regression task ===
    if (tasks is None or "regression" in tasks) and reg_label is not None:
        reg_models = get_reg_models(args.seed)

        y_all = to_diopters(df_train[reg_label]).values
        y_eval_all = to_diopters(df_eval[reg_label]).values

        mask_train = ~np.isnan(y_all)
        mask_eval = ~np.isnan(y_eval_all)

        mask_train_split = mask_train[train_idx]
        mask_test_split = mask_train[test_idx]

        y_train = y_all[train_idx][mask_train_split]
        y_test = y_all[test_idx][mask_test_split]

        X_train_task = X_train.iloc[mask_train_split]
        X_test_task = X_test.iloc[mask_test_split]

        y_eval = y_eval_all[mask_eval]
        X_eval_task = X_eval.iloc[mask_eval]

        for model_name, model in reg_models.items():
            if run_models is not None and normalize_model_name(model_name) not in run_models:
                continue
            model_key = normalize_model_name(model_name)
            tune_this = (
                bool(args.tune)
                and model_has_search_space(model_name)
                and (tune_models is None or model_key in tune_models)
            )
            tuned_params: Optional[Dict[str, Any]] = None
            loaded_params = (
                best_params_by_task_model.get(("regression", model_name)) if best_params_by_task_model else None
            )
            if loaded_params:
                tuned_params = loaded_params
            if tune_this:
                cache_key = ("regression", model_name)
                if cache_key not in tuned_cache:
                    best_json_path = paths["tuning"] / f"regression__{slugify(model_name)}.best.json"
                    if args.resume:
                        existing = load_best_json(best_json_path)
                        existing_params = (existing or {}).get("best_params")
                        if isinstance(existing, dict) and existing.get("skipped") is True:
                            tuned_cache[cache_key] = {}
                            tuning_records.append(existing)
                        elif isinstance(existing_params, dict) and existing_params:
                            tuned_cache[cache_key] = existing_params
                            tuning_records.append(existing)  # type: ignore[arg-type]

                if tune_this and cache_key not in tuned_cache:
                    cv_used = safe_cv_splits_regression(y_train, args.tune_cv)
                    n_samples = int(len(y_train))
                    es_rounds = (
                        int(args.tune_early_stopping_rounds)
                        if (is_xgboost_model(model_name) or is_lightgbm_model(model_name))
                        else 0
                    )
                    max_est = int(args.tune_max_estimators) if es_rounds > 0 else 0

                    if cv_used < 2:
                        tuned_cache[cache_key] = {}
                        record = {
                            "task": "regression",
                            "task_type": "regression",
                            "model": model_name,
                            "scoring": "r2",
                            "cv": cv_used,
                            "n_trials": args.tune_trials,
                            "n_samples": n_samples,
                            "skipped": True,
                            "reason": f"Not enough samples for CV (n={n_samples}).",
                        }
                        tuning_records.append(record)
                        save_json(best_json_path, record)
                    else:
                        try:
                            tuned_params, tuned_cv, study = tune_model_optuna(
                                base_model=model,
                                model_name=model_name,
                                task_type="regression",
                                X_train=X_train_task,
                                y_train=y_train,
                                preprocessor=preprocessor,
                                scoring="r2",
                                n_trials=args.tune_trials,
                                cv=cv_used,
                                seed=args.seed,
                                n_jobs=args.tune_jobs,
                                timeout=args.tune_timeout,
                                early_stopping_rounds=es_rounds,
                                max_estimators=max_est,
                            )
                            tuned_cache[cache_key] = tuned_params

                            record = {
                                "task": "regression",
                                "task_type": "regression",
                                "model": model_name,
                                "scoring": "r2",
                                "cv": cv_used,
                                "n_trials": args.tune_trials,
                                "n_samples": n_samples,
                                "early_stopping_rounds": es_rounds,
                                "max_estimators": max_est,
                                "best_value": tuned_cv,
                                "best_params": tuned_params,
                            }
                            tuning_records.append(record)
                            save_json(best_json_path, record)
                            try:
                                study_df = study.trials_dataframe()  # type: ignore[attr-defined]
                                study_df.to_csv(
                                    paths["tuning"] / f"regression__{slugify(model_name)}.trials.csv",
                                    index=False,
                                )
                            except Exception:
                                pass
                        except Exception as e:
                            tuned_cache[cache_key] = {}
                            record = {
                                "task": "regression",
                                "task_type": "regression",
                                "model": model_name,
                                "scoring": "r2",
                                "cv": cv_used,
                                "n_trials": args.tune_trials,
                                "n_samples": n_samples,
                                "early_stopping_rounds": es_rounds,
                                "max_estimators": max_est,
                                "skipped": True,
                                "reason": str(e),
                            }
                            tuning_records.append(record)
                            save_json(best_json_path, record)
                tuned_params = tuned_cache[cache_key] or None

            tuned_model = clone(model)
            if tuned_params:
                tuned_model.set_params(**tuned_params)
            pipe = Pipeline([("prep", clone(preprocessor)), ("model", tuned_model)])
            pipe.fit(X_train_task, y_train)

            pred_int = pipe.predict(X_test_task)
            pred_ext = pipe.predict(X_eval_task)

            int_m = regression_metrics(y_test, pred_int)
            ext_m = regression_metrics(y_eval, pred_ext)

            def r2_fn(yt, yp, yp_prob):
                return r2_score(yt, yp)

            def mae_fn(yt, yp, yp_prob):
                return mean_absolute_error(yt, yp)

            def rmse_fn(yt, yp, yp_prob):
                return math.sqrt(mean_squared_error(yt, yp))

            ci = {}
            if args.n_bootstrap > 0:
                ci_int = bootstrap_regression_cis(rng, y_test, pred_int, args.n_bootstrap)
                ci_ext = bootstrap_regression_cis(rng, y_eval, pred_ext, args.n_bootstrap)
                for key in ["r2", "mae", "rmse"]:
                    ci[f"{key}_int_lower"], ci[f"{key}_int_upper"] = ci_int.get(key, (math.nan, math.nan))
                    ci[f"{key}_ext_lower"], ci[f"{key}_ext_upper"] = ci_ext.get(key, (math.nan, math.nan))

            model_path = paths["models"] / f"regression__{model_name}.pkl"
            joblib.dump(pipe, model_path)

            pred_dir = paths["predictions"] / "regression"
            pred_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"y_true": y_test, "y_pred": pred_int}).to_csv(
                pred_dir / f"{model_name}__internal.csv", index=False
            )
            pd.DataFrame({"y_true": y_eval, "y_pred": pred_ext}).to_csv(
                pred_dir / f"{model_name}__external.csv", index=False
            )

            metrics_rows.append(
                {
                    "Model": model_name,
                    "Task": "regression",
                    "Split": "both",
                    **{f"{k}_int": v for k, v in int_m.items()},
                    **{f"{k}_ext": v for k, v in ext_m.items()},
                    **ci,
                }
            )

    # Write outputs
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = paths["metrics"] / ("metrics.csv" if tasks_tag == "all" else f"metrics_{tasks_tag}.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    metrics_json = paths["metrics"] / ("metrics.json" if tasks_tag == "all" else f"metrics_{tasks_tag}.json")
    save_json(metrics_json, {"metrics": metrics_rows})
    if tuning_records:
        tuning_json = paths["metrics"] / (
            "tuning_summary.json" if tasks_tag == "all" else f"tuning_summary_{tasks_tag}.json"
        )
        tuning_csv = paths["metrics"] / (
            "tuning_summary.csv" if tasks_tag == "all" else f"tuning_summary_{tasks_tag}.csv"
        )
        save_json(tuning_json, {"tuning": tuning_records})
        pd.DataFrame(tuning_records).to_csv(tuning_csv, index=False)
        hp_rows = []
        for rec in tuning_records:
            params = rec.get("best_params") or {}
            for k, v in params.items():
                hp_rows.append(
                    {
                        "Task": rec.get("task"),
                        "TaskType": rec.get("task_type"),
                        "Model": rec.get("model"),
                        "Hyperparameter": k,
                        "OptimalValue": v,
                    }
                )
        hyperparams_csv = paths["metrics"] / ("hyperparams.csv" if tasks_tag == "all" else f"hyperparams_{tasks_tag}.csv")
        pd.DataFrame(hp_rows).to_csv(hyperparams_csv, index=False)

        def _get_best_params(task: str, task_type: str, model: str) -> Dict[str, Any]:
            for rec in tuning_records:
                if rec.get("task") == task and rec.get("task_type") == task_type and rec.get("model") == model:
                    return rec.get("best_params") or {}
            return {}

        def _write_required_table(task: str, out_path: Path) -> None:
            spec = [
                (
                    "Random Forest",
                    [("n_estimators", "n_estimators"), ("max_depth", "max_depth"), ("min_samples_split", "min_samples_split")],
                ),
                (
                    "Extra Trees",
                    [("n_estimators", "n_estimators"), ("max_depth", "max_depth"), ("min_samples_split", "min_samples_split")],
                ),
                (
                    "Gradient Boosting",
                    [("learning_rate", "learning_rate"), ("n_estimators", "n_estimators"), ("max_depth", "max_depth")],
                ),
                (
                    "Decision Tree",
                    [("max_depth", "max_depth"), ("min_samples_split", "min_samples_split"), ("criterion", "criterion")],
                ),
                (
                    "K-Nearest Neighbors",
                    [("n_neighbors", "n_neighbors"), ("weights", "weights"), ("p (Metric)", "p")],
                ),
                (
                    "Logistic Regression",
                    [("C (Regularization)", "C"), ("class_weight", "class_weight"), ("solver", "solver")],
                ),
                ("Naive Bayes", [("var_smoothing", "var_smoothing"), ("-", None), ("-", None)]),
            ]
            rows = []
            for model_name, params_spec in spec:
                best = _get_best_params(task, "classification", model_name)
                row: Dict[str, Any] = {"Model Type": "Machine Learning", "Model Name": model_name}
                for i, (label, key) in enumerate(params_spec, start=1):
                    row[f"Hyperparameter {i}"] = label
                    if key is None:
                        row[f"Optimal Value {i}"] = "-"
                    else:
                        value = best.get(key, "-")
                        if value is None:
                            value = "None"
                        row[f"Optimal Value {i}"] = value
                rows.append(row)
            pd.DataFrame(rows).to_csv(out_path, index=False)

        def _write_required_table_regression(out_path: Path) -> None:
            spec = [
                (
                    "Random Forest",
                    [("n_estimators", "n_estimators"), ("max_depth", "max_depth"), ("min_samples_split", "min_samples_split")],
                ),
                (
                    "Extra Trees",
                    [("n_estimators", "n_estimators"), ("max_depth", "max_depth"), ("min_samples_split", "min_samples_split")],
                ),
                (
                    "Gradient Boosting",
                    [("learning_rate", "learning_rate"), ("n_estimators", "n_estimators"), ("max_depth", "max_depth")],
                ),
                (
                    "Decision Tree",
                    [("max_depth", "max_depth"), ("min_samples_split", "min_samples_split"), ("criterion", "criterion")],
                ),
                (
                    "K-Nearest Neighbors",
                    [("n_neighbors", "n_neighbors"), ("weights", "weights"), ("p (Metric)", "p")],
                ),
            ]
            rows = []
            for model_name, params_spec in spec:
                best = _get_best_params("regression", "regression", model_name)
                row: Dict[str, Any] = {"Model Type": "Machine Learning", "Model Name": model_name}
                for i, (label, key) in enumerate(params_spec, start=1):
                    row[f"Hyperparameter {i}"] = label
                    value = best.get(key, "-") if key else "-"
                    if value is None:
                        value = "None"
                    row[f"Optimal Value {i}"] = value
                rows.append(row)
            pd.DataFrame(rows).to_csv(out_path, index=False)

        # Tables for paper filling (classification tasks).
        if tasks is None or "myopia" in tasks:
            _write_required_table("myopia", paths["metrics"] / "hyperparam_table_myopia.csv")
        if tasks is None or "fast_progress" in tasks:
            _write_required_table("fast_progress", paths["metrics"] / "hyperparam_table_fast_progress.csv")
        if tasks is None or "regression" in tasks:
            _write_required_table_regression(paths["metrics"] / "hyperparam_table_regression.csv")

    (paths["logs"] / "run.log").write_text(
        f"started_at={start_time}\nfinished_at={datetime.now().isoformat()}\n",
        encoding="utf-8",
    )

    print(f"[OK] wrote: {metrics_csv}")


if __name__ == "__main__":
    main()
