import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_CREATE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    r2_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

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

MEASURE_COLS = [
    "baseline_sphere_self",
    "baseline_cylinder_self",
    "baseline_axis_self",
    "baseline_sphere_other",
    "baseline_cylinder_other",
    "baseline_axis_other",
]

CAT_FEATURE_CANDIDATES = [
    "sex",
    "gender",
    "intervention_method",
    "lens_function_left",
    "lens_function_right",
]
DEFAULT_SEED = 42

# Splits, tuning folds and bootstrap resamples are done at the patient level.
GROUP_ID_CANDIDATES = ["patient_id", "subject_id", "person_id"]

YES_TOKENS = {"yes", "y", "true", "1"}
NO_TOKENS = {"no", "n", "false", "0"}


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


def resolve_group_ids(
    df: pd.DataFrame, group_col: Optional[str], *, required: bool, context: str
) -> Optional[np.ndarray]:
    col = group_col
    if not col:
        col = next((c for c in GROUP_ID_CANDIDATES if c in df.columns), None)
    if col is None or col not in df.columns:
        if required:
            raise ValueError(
                f"{context}: missing patient ID column (tried {group_col or GROUP_ID_CANDIDATES}). "
                "Provide --group-col (or pass --allow-row-split to split by row)."
            )
        return None
    if df[col].isna().any():
        raise ValueError(f"{context}: patient ID column `{col}` contains missing values.")
    return df[col].astype(str).str.strip().to_numpy()


def patient_train_test_split(
    indices: np.ndarray,
    groups: Optional[np.ndarray],
    test_size: float,
    seed: int,
    stratify: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split `indices` so that no patient appears on both sides. `groups`/`stratify` are indexed like the full data."""
    if groups is None:
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify[indices] if stratify is not None else None,
        )
    sub_groups = groups[indices]
    if stratify is not None:
        n_splits = max(2, int(round(1.0 / test_size)))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        tr, te = next(splitter.split(indices, stratify[indices], sub_groups))
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(splitter.split(indices, groups=sub_groups))
    overlap = set(sub_groups[tr]) & set(sub_groups[te])
    assert not overlap, f"{len(overlap)} patients appear in both train and test"
    return np.sort(indices[tr]), np.sort(indices[te])


def make_bootstrap_sampler(rng: np.random.Generator, n: int, groups: Optional[np.ndarray]):
    """Row bootstrap if groups is None, otherwise cluster bootstrap that resamples whole patients."""
    if groups is None:
        return lambda: rng.integers(0, n, size=n)
    _, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    n_groups = len(counts)

    def draw() -> np.ndarray:
        picked = rng.integers(0, n_groups, size=n_groups)
        lengths = counts[picked]
        offsets = np.arange(int(lengths.sum())) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        return order[np.repeat(starts[picked], lengths) + offsets]

    return draw


def extract_binary_labels(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    normalized = series.astype(str).str.strip().str.lower()
    mask = normalized.isin(YES_TOKENS | NO_TOKENS).to_numpy()
    y = (normalized[mask].isin(YES_TOKENS)).astype(np.float32).to_numpy()
    return mask, y


def to_diopters(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mask = values.abs() > 50
    if mask.any():
        values.loc[mask] = values.loc[mask] / 100.0
    return values


def resolve_regression_label(df_train: pd.DataFrame, df_eval: Optional[pd.DataFrame]) -> Optional[str]:
    for col in REGRESSION_LABEL_CANDIDATES:
        if col in df_train.columns and (df_eval is None or col in df_eval.columns):
            return col
    return None


class TabDataset(Dataset):
    def __init__(self, num_x: np.ndarray, cat_x: np.ndarray, y: np.ndarray):
        self.num_x = torch.from_numpy(num_x.astype(np.float32))
        if cat_x.size == 0:
            self.cat_x = torch.zeros((len(self.num_x), 0), dtype=torch.int64)
        else:
            self.cat_x = torch.from_numpy(cat_x.astype(np.int64))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.num_x[idx], self.cat_x[idx], self.y[idx]


class FTTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        cat_cardinalities: List[int],
        d_token: int = 64,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        ff_factor: int = 4,
        out_dim: int = 1,
    ):
        super().__init__()
        self.num_mlps = nn.ModuleList([nn.Linear(1, d_token) for _ in range(num_features)])
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_token) for cardinality in cat_cardinalities]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * ff_factor,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, out_dim),
        )

    def forward(self, num_x: torch.Tensor, cat_x: torch.Tensor) -> torch.Tensor:
        tokens = []
        for i, layer in enumerate(self.num_mlps):
            tokens.append(layer(num_x[:, i : i + 1]))
        for i, emb in enumerate(self.cat_embeddings):
            tokens.append(emb(cat_x[:, i]))
        x = torch.stack(tokens, dim=1)
        cls = self.cls_token.repeat(num_x.size(0), 1, 1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        cls_out = self.norm(x[:, 0])
        return self.head(cls_out).squeeze(-1)


def ensure_age_column(df: pd.DataFrame) -> str:
    for candidate in ["age_baseline", "age"]:
        if candidate in df.columns:
            df[candidate] = pd.to_numeric(df[candidate], errors="coerce")
            return candidate
    raise ValueError("Missing required age column: expected `age` or `age_baseline`.")


def ensure_distance_column(df: pd.DataFrame) -> str:
    for candidate in [STANDARD_DAYS_COLUMN, "followup_days", "days", "time_days"]:
        if candidate in df.columns:
            df[STANDARD_DAYS_COLUMN] = (
                df[candidate].astype(str).str.extract(r"(-?\d+(?:\.\d+)?)")[0].astype(float)
            )
            return STANDARD_DAYS_COLUMN
    raise ValueError(
        "Missing required follow-up days column: expected `days_since_first_test` "
        "(or `followup_days`/`days`/`time_days`)."
    )


def encode_categories(
    df: pd.DataFrame,
    cat_feature_names: List[str],
    existing_maps: Optional[Dict[str, Dict[str, int]]] = None,
) -> (np.ndarray, Dict[str, Dict[str, int]]):
    mappings = {} if existing_maps is None else existing_maps.copy()
    arrays = []
    for feature in cat_feature_names:
        series = df[feature].fillna(UNKNOWN_CATEGORY).astype(str)
        if existing_maps is None:
            classes = sorted(series.unique().tolist())
            mapping = {cls: idx for idx, cls in enumerate(classes)}
            mappings[feature] = mapping
        else:
            mapping = mappings.get(feature, {})
        encoded = series.map(lambda x: mapping.get(x, 0)).astype(int).to_numpy()
        arrays.append(encoded)
    if arrays:
        cat_data = np.stack(arrays, axis=1)
    else:
        cat_data = np.zeros((len(df), 0), dtype=np.int64)
    return cat_data, mappings


def prepare_eval_context(
    df: pd.DataFrame,
    num_feature_names: List[str],
    cat_feature_names: List[str],
    num_medians: pd.Series,
    scaler: StandardScaler,
    cat_maps: Dict[str, Dict[str, int]],
    reg_task_col: Optional[str],
    groups: Optional[np.ndarray] = None,
) -> Dict:
    num_df = df[num_feature_names].copy()
    num_df = num_df.fillna(num_medians)
    num_data = scaler.transform(num_df.values)
    cat_data, _ = encode_categories(df, cat_feature_names, cat_maps)

    class_context = {}
    for key, col in CLASS_TASKS.items():
        if col not in df.columns:
            continue
        mask, y = extract_binary_labels(df[col])
        if not mask.any():
            continue
        class_context[key] = {
            "num": num_data[mask],
            "cat": cat_data[mask],
            "y": y,
            "groups": groups[mask] if groups is not None else None,
        }

    reg_context = None
    if reg_task_col and reg_task_col in df.columns:
        values = to_diopters(df[reg_task_col]).values
        mask = ~np.isnan(values)
        if mask.any():
            reg_context = {
                "num": num_data[mask],
                "cat": cat_data[mask],
                "y": values[mask].astype(np.float32),
                "groups": groups[mask] if groups is not None else None,
            }
    return {
        "class": class_context,
        "reg": reg_context,
    }


def save_ft_model(
    model: FTTransformer,
    path: Path,
    num_feature_names: List[str],
    cat_feature_names: List[str],
    scaler: StandardScaler,
    num_medians: pd.Series,
    cat_maps: Dict[str, Dict[str, int]],
):
    payload = {
        "state_dict": model.state_dict(),
        "num_feature_names": num_feature_names,
        "cat_feature_names": cat_feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "num_medians": {k: float(v) for k, v in num_medians.items()},
        "cat_maps": cat_maps,
    }
    torch.save(payload, path)


def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    preds_all, targets_all = [], []
    for num_x, cat_x, y in loader:
        num_x = num_x.to(device)
        cat_x = cat_x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train):
            preds = model(num_x, cat_x)
            loss = criterion(preds, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(y)
        preds_all.append(preds.detach().cpu().numpy())
        targets_all.append(y.detach().cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    preds = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)
    return avg_loss, preds, targets


def _bootstrap_ci(values: np.ndarray, alpha: float = 0.05) -> (float, float):
    if values.size == 0:
        return math.nan, math.nan
    lower = float(np.nanpercentile(values, 100 * (alpha / 2)))
    upper = float(np.nanpercentile(values, 100 * (1 - alpha / 2)))
    return lower, upper


def _bootstrap_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int,
    seed: int,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    sample = make_bootstrap_sampler(rng, n, groups)
    aucs, accs, f1s, senss, specs = [], [], [], [], []
    for _ in range(n_bootstrap):
        idx = sample()
        yt = y_true[idx]
        yp = y_prob[idx]
        yhat = (yp >= 0.5).astype(int)
        try:
            aucs.append(roc_auc_score(yt, yp))
        except ValueError:
            aucs.append(np.nan)
        accs.append(accuracy_score(yt, yhat))
        f1s.append(f1_score(yt, yhat, zero_division=0))
        senss.append(recall_score(yt, yhat, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(yt, yhat, labels=[0, 1]).ravel()
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else math.nan)

    auc_lower, auc_upper = _bootstrap_ci(np.array(aucs))
    acc_lower, acc_upper = _bootstrap_ci(np.array(accs))
    f1_lower, f1_upper = _bootstrap_ci(np.array(f1s))
    sens_lower, sens_upper = _bootstrap_ci(np.array(senss))
    spec_lower, spec_upper = _bootstrap_ci(np.array(specs))
    return {
        "auc_lower": auc_lower,
        "auc_upper": auc_upper,
        "acc_lower": acc_lower,
        "acc_upper": acc_upper,
        "f1_lower": f1_lower,
        "f1_upper": f1_upper,
        "sens_lower": sens_lower,
        "sens_upper": sens_upper,
        "spec_lower": spec_lower,
        "spec_upper": spec_upper,
    }


def _bootstrap_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int,
    seed: int,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    sample = make_bootstrap_sampler(rng, n, groups)
    r2s, maes, rmses = [], [], []
    for _ in range(n_bootstrap):
        idx = sample()
        yt = y_true[idx]
        yp = y_pred[idx]
        try:
            r2s.append(r2_score(yt, yp))
        except ValueError:
            r2s.append(np.nan)
        maes.append(mean_absolute_error(yt, yp))
        rmses.append(mean_squared_error(yt, yp) ** 0.5)

    r2_lower, r2_upper = _bootstrap_ci(np.array(r2s))
    mae_lower, mae_upper = _bootstrap_ci(np.array(maes))
    rmse_lower, rmse_upper = _bootstrap_ci(np.array(rmses))
    return {
        "r2_lower": r2_lower,
        "r2_upper": r2_upper,
        "mae_lower": mae_lower,
        "mae_upper": mae_upper,
        "rmse_lower": rmse_lower,
        "rmse_upper": rmse_upper,
    }


def train_task(
    task_name: str,
    y_values: np.ndarray,
    mask: np.ndarray,
    problem_type: str,
    device: torch.device,
    num_data: np.ndarray,
    cat_data: np.ndarray,
    cat_cardinalities: List[int],
    eval_context: Optional[Dict],
    global_train_mask: np.ndarray,
    global_test_mask: np.ndarray,
    epochs: int,
    batch_size: int,
    d_token: int,
    learning_rate: float,
    weight_decay: float,
    pred_dir: Optional[Path],
    model_name: str,
    n_bootstrap: int,
    seed: int,
    groups: Optional[np.ndarray] = None,
):
    indices = np.where(mask)[0]
    if indices.size == 0:
        raise ValueError(f"No valid samples for task {task_name}.")

    train_indices, val_indices = _get_task_split_indices(
        indices=indices,
        global_train_mask=global_train_mask,
        global_test_mask=global_test_mask,
        y_values=y_values,
        problem_type=problem_type,
        seed=seed,
        groups=groups,
    )
    val_groups = groups[val_indices] if groups is not None else None

    train_dataset = TabDataset(num_data[train_indices], cat_data[train_indices], y_values[train_indices])
    val_dataset = TabDataset(num_data[val_indices], cat_data[val_indices], y_values[val_indices])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = FTTransformer(
        num_features=num_data.shape[1],
        cat_cardinalities=cat_cardinalities,
        d_token=d_token,
        n_heads=8,
        n_layers=4,
        dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss() if problem_type == "classification" else nn.MSELoss()

    for epoch in range(epochs):
        train_loss, _, _ = run_epoch(model, train_loader, optimizer, criterion, device, True)
        val_loss, _, _ = run_epoch(model, val_loader, optimizer, criterion, device, False)
        print(f"[{task_name}] epoch {epoch+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    _, val_preds, val_targets = run_epoch(model, val_loader, optimizer, criterion, device, False)
    metrics = {"Task": task_name}
    if problem_type == "classification":
        probs = torch.sigmoid(torch.from_numpy(val_preds)).numpy()
        preds = (probs >= 0.5).astype(int)
        try:
            auc = roc_auc_score(val_targets, probs)
        except ValueError:
            auc = np.nan
        acc = accuracy_score(val_targets, preds)
        f1 = f1_score(val_targets, preds, zero_division=0)
        sens = recall_score(val_targets, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(val_targets, preds, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else math.nan
        metrics.update(
            {
                "auc_int": float(auc),
                "acc_int": float(acc),
                "f1_int": float(f1),
                "sens_int": float(sens),
                "spec_int": float(spec),
            }
        )
        if n_bootstrap > 0 and len(val_targets) > 1:
            ci = _bootstrap_classification_metrics(val_targets.astype(int), probs, n_bootstrap, seed, val_groups)
            metrics.update(
                {
                    "auc_int_lower": ci["auc_lower"],
                    "auc_int_upper": ci["auc_upper"],
                    "acc_int_lower": ci["acc_lower"],
                    "acc_int_upper": ci["acc_upper"],
                    "f1_int_lower": ci["f1_lower"],
                    "f1_int_upper": ci["f1_upper"],
                    "sens_int_lower": ci["sens_lower"],
                    "sens_int_upper": ci["sens_upper"],
                    "spec_int_lower": ci["spec_lower"],
                    "spec_int_upper": ci["spec_upper"],
                }
            )
        if pred_dir is not None:
            task_dir = pred_dir / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {"y_true": val_targets.astype(int), "y_prob": probs, "y_pred": preds}
            ).to_csv(task_dir / f"{model_name}__internal.csv", index=False)
        ext = eval_context.get("class", {}).get(task_name) if eval_context else None
        if ext:
            eval_dataset = TabDataset(ext["num"], ext["cat"], ext["y"])
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
            _, eval_preds, eval_targets = run_epoch(
                model, eval_loader, optimizer, criterion, device, False
            )
            eval_probs = torch.sigmoid(torch.from_numpy(eval_preds)).numpy()
            eval_preds_bin = (eval_probs >= 0.5).astype(int)
            try:
                auc_ext = roc_auc_score(eval_targets, eval_probs)
            except ValueError:
                auc_ext = np.nan
            acc_ext = accuracy_score(eval_targets, eval_preds_bin)
            f1_ext = f1_score(eval_targets, eval_preds_bin, zero_division=0)
            sens_ext = recall_score(eval_targets, eval_preds_bin, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(eval_targets, eval_preds_bin, labels=[0, 1]).ravel()
            spec_ext = tn / (tn + fp) if (tn + fp) > 0 else math.nan
            metrics.update(
                {
                    "auc_ext": float(auc_ext),
                    "acc_ext": float(acc_ext),
                    "f1_ext": float(f1_ext),
                    "sens_ext": float(sens_ext),
                    "spec_ext": float(spec_ext),
                }
            )
            if n_bootstrap > 0 and len(eval_targets) > 1:
                ci = _bootstrap_classification_metrics(
                    eval_targets.astype(int), eval_probs, n_bootstrap, seed + 1, ext.get("groups")
                )
                metrics.update(
                    {
                        "auc_ext_lower": ci["auc_lower"],
                        "auc_ext_upper": ci["auc_upper"],
                        "acc_ext_lower": ci["acc_lower"],
                        "acc_ext_upper": ci["acc_upper"],
                        "f1_ext_lower": ci["f1_lower"],
                        "f1_ext_upper": ci["f1_upper"],
                        "sens_ext_lower": ci["sens_lower"],
                        "sens_ext_upper": ci["sens_upper"],
                        "spec_ext_lower": ci["spec_lower"],
                        "spec_ext_upper": ci["spec_upper"],
                    }
                )
            if pred_dir is not None:
                task_dir = pred_dir / task_name
                task_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {"y_true": eval_targets.astype(int), "y_prob": eval_probs, "y_pred": eval_preds_bin}
                ).to_csv(task_dir / f"{model_name}__external.csv", index=False)
    else:
        r2 = r2_score(val_targets, val_preds)
        mae = mean_absolute_error(val_targets, val_preds)
        rmse = mean_squared_error(val_targets, val_preds) ** 0.5
        metrics.update({"r2_int": float(r2), "mae_int": float(mae), "rmse_int": float(rmse)})
        if n_bootstrap > 0 and len(val_targets) > 1:
            ci = _bootstrap_regression_metrics(val_targets, val_preds, n_bootstrap, seed, val_groups)
            metrics.update(
                {
                    "r2_int_lower": ci["r2_lower"],
                    "r2_int_upper": ci["r2_upper"],
                    "mae_int_lower": ci["mae_lower"],
                    "mae_int_upper": ci["mae_upper"],
                    "rmse_int_lower": ci["rmse_lower"],
                    "rmse_int_upper": ci["rmse_upper"],
                }
            )
        if pred_dir is not None:
            task_dir = pred_dir / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"y_true": val_targets, "y_pred": val_preds}).to_csv(
                task_dir / f"{model_name}__internal.csv", index=False
            )
        ext = eval_context.get("reg") if eval_context else None
        if ext:
            eval_dataset = TabDataset(ext["num"], ext["cat"], ext["y"])
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
            _, eval_preds, eval_targets = run_epoch(
                model, eval_loader, optimizer, criterion, device, False
            )
            metrics.update(
                {
                    "r2_ext": float(r2_score(eval_targets, eval_preds)),
                    "mae_ext": float(mean_absolute_error(eval_targets, eval_preds)),
                    "rmse_ext": float(mean_squared_error(eval_targets, eval_preds) ** 0.5),
                }
            )
            if n_bootstrap > 0 and len(eval_targets) > 1:
                ci = _bootstrap_regression_metrics(eval_targets, eval_preds, n_bootstrap, seed + 1, ext.get("groups"))
                metrics.update(
                    {
                        "r2_ext_lower": ci["r2_lower"],
                        "r2_ext_upper": ci["r2_upper"],
                        "mae_ext_lower": ci["mae_lower"],
                        "mae_ext_upper": ci["mae_upper"],
                        "rmse_ext_lower": ci["rmse_lower"],
                        "rmse_ext_upper": ci["rmse_upper"],
                    }
                )
            if pred_dir is not None:
                task_dir = pred_dir / task_name
                task_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"y_true": eval_targets, "y_pred": eval_preds}).to_csv(
                    task_dir / f"{model_name}__external.csv", index=False
                )
    return metrics, model


def _get_task_split_indices(
    *,
    indices: np.ndarray,
    global_train_mask: np.ndarray,
    global_test_mask: np.ndarray,
    y_values: np.ndarray,
    problem_type: str,
    seed: int,
    groups: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    # Prefer the global patient-level split (same across tasks), with a fallback if it becomes degenerate.
    train_indices = indices[global_train_mask[indices]]
    val_indices = indices[global_test_mask[indices]]
    if train_indices.size == 0 or val_indices.size == 0:
        train_indices, val_indices = patient_train_test_split(indices, groups, 0.2, seed)
    if problem_type == "classification":
        y_train = y_values[train_indices]
        y_val = y_values[val_indices]
        if (np.unique(y_train).size < 2) or (np.unique(y_val).size < 2):
            stratify = y_values if np.unique(y_values[indices]).size > 1 else None
            train_indices, val_indices = patient_train_test_split(indices, groups, 0.2, seed, stratify)
    return train_indices, val_indices


def tune_ft_transformer(
    *,
    task_name: str,
    problem_type: str,
    y_values: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    num_data: np.ndarray,
    cat_data: np.ndarray,
    cat_cardinalities: List[int],
    global_train_mask: np.ndarray,
    global_test_mask: np.ndarray,
    trials: int,
    epochs: int,
    batch_size: int,
    max_samples: Optional[int],
    seed: int,
    cv: int = 5,
    groups: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], float, Any]:
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna is required for --tune. Install with: pip install optuna")

    indices = np.where(mask)[0]
    if indices.size == 0:
        raise ValueError(f"No valid samples for tuning task {task_name}.")

    # Tune only on the training portion; the internal test split is never seen during tuning.
    pool = indices[global_train_mask[indices]]
    if pool.size == 0:
        raise ValueError(f"No training samples for tuning task {task_name}.")
    pool_groups = groups[pool] if groups is not None else None
    if cv < 2:
        raise ValueError("--tune-cv must be >= 2.")
    if problem_type == "classification":
        if pool_groups is not None:
            splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=seed)
        else:
            from sklearn.model_selection import StratifiedKFold

            splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        split_y = y_values[pool].astype(int)
    else:
        if pool_groups is not None:
            splitter = GroupKFold(n_splits=cv)
        else:
            from sklearn.model_selection import KFold

            splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
        split_y = None

    rng = np.random.default_rng(seed)
    max_samples_int = int(max_samples) if max_samples is not None and int(max_samples) > 0 else None
    fold_loaders = []
    for fold_tr, fold_val in splitter.split(pool, split_y, pool_groups):
        train_indices, val_indices = pool[fold_tr], pool[fold_val]
        if max_samples_int is not None:
            if train_indices.size > max_samples_int:
                train_indices = rng.choice(train_indices, size=max_samples_int, replace=False)
            if val_indices.size > max_samples_int:
                val_indices = rng.choice(val_indices, size=max_samples_int, replace=False)
        train_dataset = TabDataset(num_data[train_indices], cat_data[train_indices], y_values[train_indices])
        val_dataset = TabDataset(num_data[val_indices], cat_data[val_indices], y_values[val_indices])
        fold_loaders.append(
            (
                DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            )
        )

    sampler = TPESampler(seed=seed)  # type: ignore[misc]
    study = optuna.create_study(direction="maximize", sampler=sampler)  # type: ignore[union-attr]

    def objective(trial: Any) -> float:
        torch.manual_seed(seed + int(trial.number))
        np.random.seed(seed + int(trial.number))

        d_token = int(trial.suggest_categorical("d_token", [32, 64, 128]))
        learning_rate = float(trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True))
        weight_decay = float(trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True))

        scores = []
        for train_loader, val_loader in fold_loaders:
            model = FTTransformer(
                num_features=num_data.shape[1],
                cat_cardinalities=cat_cardinalities,
                d_token=d_token,
                n_heads=8,
                n_layers=4,
                dropout=0.1,
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = nn.BCEWithLogitsLoss() if problem_type == "classification" else nn.MSELoss()

            for _ in range(int(epochs)):
                run_epoch(model, train_loader, optimizer, criterion, device, True)

            _, val_preds, val_targets = run_epoch(model, val_loader, optimizer, criterion, device, False)
            if problem_type == "classification":
                probs = torch.sigmoid(torch.from_numpy(val_preds)).numpy()
                if len(np.unique(val_targets)) < 2:
                    return -1e9
                scores.append(float(roc_auc_score(val_targets, probs)))
            else:
                scores.append(float(r2_score(val_targets, val_preds)))
        return float(np.mean(scores))

    study.optimize(objective, n_trials=trials, show_progress_bar=False)  # type: ignore[union-attr]
    return dict(study.best_params), float(study.best_value), study


def main():
    parser = argparse.ArgumentParser(description="Train FT-Transformer on tabular dataset.")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to training CSV dataset.",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=None,
        help="Optional CSV for external evaluation.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="JSON metrics output path.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="CSV metrics output path.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models_ft"),
        help="Directory to store trained FT-Transformer models.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--group-col",
        type=str,
        default=None,
        help=(
            "Patient ID column used for the train/test split, tuning folds and cluster bootstrap "
            f"(default: first present of {GROUP_ID_CANDIDATES})."
        ),
    )
    parser.add_argument(
        "--allow-row-split",
        action="store_true",
        help="Fall back to row-level splitting when no patient ID column exists.",
    )
    parser.add_argument("--d-token", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated tasks to train: myopia,fast_progress,regression (default: all).",
    )
    parser.add_argument(
        "--column-map",
        type=Path,
        default=None,
        help="Optional JSON file mapping input column names to canonical names expected by this script.",
    )
    parser.add_argument(
        "--load-best-dir",
        type=Path,
        default=None,
        help="Directory containing ft_transformer__<task>.best.json to load hyperparameters for final training.",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help="Optional directory to save predictions CSVs (internal/external) per task.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Use Optuna to tune (learning_rate, weight_decay, d_token) before training.",
    )
    parser.add_argument("--tune-task", choices=["myopia", "fast_progress", "regression"], default="myopia")
    parser.add_argument(
        "--tune-tasks",
        type=str,
        default=None,
        help="Comma-separated tuning tasks: myopia,fast_progress,regression (overrides --tune-task).",
    )
    parser.add_argument(
        "--tune-only",
        action="store_true",
        help="Only run tuning and write tuning artifacts; skip final model training/eval.",
    )
    parser.add_argument("--tune-trials", type=int, default=30)
    parser.add_argument(
        "--tune-cv",
        type=int,
        default=5,
        help="Patient-grouped K-fold CV on the internal training split used for tuning (test split is not used).",
    )
    parser.add_argument("--tune-epochs", type=int, default=10)
    parser.add_argument("--tune-batch-size", type=int, default=None)
    parser.add_argument(
        "--tune-max-samples",
        type=int,
        default=None,
        help="Optional cap on train/val samples per tuning task to speed up tuning (applies after split).",
    )
    args = parser.parse_args()

    selected_tasks = None
    if str(args.tasks).strip().lower() != "all":
        selected_tasks = {t.strip() for t in str(args.tasks).split(",") if t.strip()}
        allowed = set(CLASS_TASKS.keys()) | {"regression"}
        unknown = sorted([t for t in selected_tasks if t not in allowed])
        if unknown:
            raise ValueError(f"Unknown --tasks: {unknown}. Allowed: {sorted(allowed)}")

    def load_best_hparams(task_name: str) -> Dict[str, Any]:
        if not args.load_best_dir:
            return {}
        best_path = args.load_best_dir / f"ft_transformer__{task_name}.best.json"
        try:
            payload = json.loads(best_path.read_text(encoding="utf-8"))
            best_params = (payload or {}).get("best_params") or {}
            return best_params if isinstance(best_params, dict) else {}
        except Exception:
            return {}

    column_map = load_column_map(args.column_map)
    df = apply_column_map(pd.read_csv(args.data), column_map)
    age_col = ensure_age_column(df)
    distance_col = ensure_distance_column(df)
    for col in MEASURE_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required feature column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    cat_feature_names = [col for col in CAT_FEATURE_CANDIDATES if col in df.columns]
    for col in cat_feature_names:
        df[col] = df[col].fillna(UNKNOWN_CATEGORY).astype(str)

    groups = resolve_group_ids(df, args.group_col, required=not args.allow_row_split, context="--data")
    if groups is None:
        print("[WARN] No patient ID column: using row-level split/bootstrap.")
    train_idx, test_idx = patient_train_test_split(np.arange(len(df)), groups, args.test_size, args.seed)
    global_train_mask = np.zeros(len(df), dtype=bool)
    global_test_mask = np.zeros(len(df), dtype=bool)
    global_train_mask[train_idx] = True
    global_test_mask[test_idx] = True
    if groups is not None:
        print(
            f"[INFO] Patient-level split: train {np.unique(groups[train_idx]).size} patients / {len(train_idx)} rows, "
            f"test {np.unique(groups[test_idx]).size} patients / {len(test_idx)} rows"
        )

    num_feature_names = [age_col, distance_col] + MEASURE_COLS
    num_df = df[num_feature_names].copy()
    num_medians = num_df.iloc[train_idx].median()
    num_df = num_df.fillna(num_medians)
    scaler = StandardScaler()
    scaler.fit(num_df.values[train_idx])
    num_data = scaler.transform(num_df.values)
    cat_data, cat_maps = encode_categories(df, cat_feature_names)
    cat_cardinalities = [len(cat_maps[feature]) for feature in cat_feature_names]

    eval_context = {}
    eval_df = None
    if args.eval_data:
        eval_df = apply_column_map(pd.read_csv(args.eval_data), column_map)
        ensure_age_column(eval_df)
        ensure_distance_column(eval_df)
        for col in MEASURE_COLS:
            if col not in eval_df.columns:
                raise ValueError(f"Eval data missing column {col}")
            eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce")
        for col in cat_feature_names:
            if col not in eval_df.columns:
                eval_df[col] = UNKNOWN_CATEGORY
            eval_df[col] = eval_df[col].fillna(UNKNOWN_CATEGORY).astype(str)
    reg_label = resolve_regression_label(df, eval_df)
    if eval_df is not None:
        eval_context = prepare_eval_context(
            eval_df,
            num_feature_names,
            cat_feature_names,
            num_medians,
            scaler,
            cat_maps,
            reg_label,
            groups=resolve_group_ids(eval_df, args.group_col, required=False, context="--eval-data"),
        )

    out_dir = args.json_output.parent if args.json_output else args.model_dir.parent
    pred_dir = args.pred_dir or (out_dir / "predictions")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    results = []
    if args.tune:
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna is required for --tune. Install with: pip install optuna")

        tune_batch_size = int(args.tune_batch_size) if args.tune_batch_size else int(args.batch_size)

        out_dir = (args.json_output.parent if args.json_output else args.model_dir.parent)
        tune_dir = out_dir / "tuning"
        tune_dir.mkdir(parents=True, exist_ok=True)

        tune_tasks = (
            [t.strip() for t in str(args.tune_tasks).split(",") if t.strip()]
            if args.tune_tasks
            else [args.tune_task]
        )
        allowed = {"myopia", "fast_progress", "regression"}
        unknown = sorted([t for t in tune_tasks if t not in allowed])
        if unknown:
            raise ValueError(f"Unknown --tune-tasks: {unknown}. Allowed: {sorted(allowed)}")
        if len(tune_tasks) > 1 and not args.tune_only:
            raise ValueError("When tuning multiple tasks, use --tune-only to avoid ambiguous training hyperparameters.")

        tune_summary_rows: List[Dict[str, Any]] = []
        for tune_task_key in tune_tasks:
            if tune_task_key == "regression":
                if reg_label is None or reg_label not in df.columns:
                    raise ValueError("Missing regression label column (no usable regression label found).")
                raw = to_diopters(df[reg_label]).values.astype(np.float32)
                mask_vals = ~np.isnan(raw)
                y_vals = np.nan_to_num(raw, nan=0.0).astype(np.float32)
                problem_type = "regression"
            else:
                label_col = CLASS_TASKS[tune_task_key]
                if label_col not in df.columns:
                    raise ValueError(f"Missing label column: {label_col}")
                mask_vals, y_masked = extract_binary_labels(df[label_col])
                y_vals = np.zeros(len(df), dtype=np.float32)
                y_vals[mask_vals] = y_masked
                problem_type = "classification"

            best_params, best_value, study = tune_ft_transformer(
                task_name=tune_task_key,
                problem_type=problem_type,
                y_values=y_vals,
                mask=mask_vals,
                device=device,
                num_data=num_data,
                cat_data=cat_data,
                cat_cardinalities=cat_cardinalities,
                global_train_mask=global_train_mask,
                global_test_mask=global_test_mask,
                trials=int(args.tune_trials),
                epochs=int(args.tune_epochs),
                batch_size=tune_batch_size,
                max_samples=args.tune_max_samples,
                seed=int(args.seed),
                cv=int(args.tune_cv),
                groups=groups,
            )

            tuned_d_token = int(best_params.get("d_token", args.d_token))
            tuned_lr = float(best_params.get("learning_rate", args.lr))
            tuned_wd = float(best_params.get("weight_decay", args.weight_decay))
            tune_metric = "roc_auc" if problem_type == "classification" else "r2"

            best_payload = {
                "model": "FT-Transformer",
                "tune_task": tune_task_key,
                "tune_metric": tune_metric,
                "best_value": best_value,
                "best_params": best_params,
            }
            (tune_dir / f"ft_transformer__{tune_task_key}.best.json").write_text(
                json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            try:
                study.trials_dataframe().to_csv(  # type: ignore[attr-defined]
                    tune_dir / f"ft_transformer__{tune_task_key}.trials.csv", index=False
                )
            except Exception:
                pass
            pd.DataFrame(
                [
                    {
                        "Task": tune_task_key,
                        "Model Type": "Deep Learning",
                        "Model Name": "FT-Transformer",
                        "Hyperparameter 1": "learning_rate",
                        "Optimal Value 1": tuned_lr,
                        "Hyperparameter 2": "weight_decay",
                        "Optimal Value 2": tuned_wd,
                        "Hyperparameter 3": "d_token",
                        "Optimal Value 3": tuned_d_token,
                        "Best Metric": tune_metric,
                        "Best Value": best_value,
                    }
                ]
            ).to_csv(tune_dir / f"ft_hyperparam_table__{tune_task_key}.csv", index=False)

            tune_summary_rows.append(
                {
                    "Task": tune_task_key,
                    "Metric": tune_metric,
                    "BestValue": best_value,
                    "learning_rate": tuned_lr,
                    "weight_decay": tuned_wd,
                    "d_token": tuned_d_token,
                }
            )

            if len(tune_tasks) == 1:
                args.d_token = tuned_d_token
                args.lr = tuned_lr
                args.weight_decay = tuned_wd

        pd.DataFrame(tune_summary_rows).to_csv(tune_dir / "ft_transformer.tuning_summary.csv", index=False)
        pd.DataFrame(
            [{"Task": r["Task"], "learning_rate": r["learning_rate"], "weight_decay": r["weight_decay"]} for r in tune_summary_rows]
        ).to_csv(tune_dir / "ft_transformer.lr_wd_by_task.csv", index=False)

        if args.tune_only:
            print(f"[OK] wrote tuning artifacts to: {tune_dir}")
            return
    for key, col in CLASS_TASKS.items():
        if selected_tasks is not None and key not in selected_tasks:
            continue
        if col not in df.columns:
            print(f"[WARN] Skip {key}: missing label column {col}")
            continue
        mask, y_masked = extract_binary_labels(df[col])
        y = np.zeros(len(df), dtype=np.float32)
        y[mask] = y_masked
        if not mask.any():
            print(f"[WARN] Skip {key}: no valid labels (yes/no) in {col}")
            continue
        best_params = load_best_hparams(key)
        d_token = int(best_params.get("d_token", args.d_token))
        learning_rate = float(best_params.get("learning_rate", args.lr))
        weight_decay = float(best_params.get("weight_decay", args.weight_decay))

        metrics, model = train_task(
            task_name=key,
            y_values=y,
            mask=mask,
            problem_type="classification",
            device=device,
            num_data=num_data,
            cat_data=cat_data,
            cat_cardinalities=cat_cardinalities,
            eval_context=eval_context,
            global_train_mask=global_train_mask,
            global_test_mask=global_test_mask,
            epochs=args.epochs,
            batch_size=args.batch_size,
            d_token=d_token,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pred_dir=pred_dir,
            model_name="FT-Transformer",
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            groups=groups,
        )
        metrics.update({"Model": "FT-Transformer", "Split": "both"})
        results.append(metrics)
        model_path = args.model_dir / f"{args.data.stem}-FT-{key}.pt"
        save_ft_model(model, model_path, num_feature_names, cat_feature_names, scaler, num_medians, cat_maps)
        print(f"[RESULT] {key}: {metrics}")

    if reg_label and reg_label in df.columns and (selected_tasks is None or "regression" in selected_tasks):
        reg_values = to_diopters(df[reg_label]).values
        reg_mask = ~np.isnan(reg_values)
        if reg_mask.any():
            best_params = load_best_hparams("regression")
            d_token = int(best_params.get("d_token", args.d_token))
            learning_rate = float(best_params.get("learning_rate", args.lr))
            weight_decay = float(best_params.get("weight_decay", args.weight_decay))
            metrics, model = train_task(
                task_name="regression",
                y_values=reg_values,
                mask=reg_mask,
                problem_type="regression",
                device=device,
                num_data=num_data,
                cat_data=cat_data,
                cat_cardinalities=cat_cardinalities,
                eval_context=eval_context,
                global_train_mask=global_train_mask,
                global_test_mask=global_test_mask,
                epochs=args.epochs,
                batch_size=args.batch_size,
                d_token=d_token,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                pred_dir=pred_dir,
                model_name="FT-Transformer",
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
                groups=groups,
            )
            metrics.update({"Model": "FT-Transformer", "Split": "both"})
            results.append(metrics)
            model_path = args.model_dir / f"{args.data.stem}-FT-regression.pt"
            save_ft_model(model, model_path, num_feature_names, cat_feature_names, scaler, num_medians, cat_maps)
            print(f"[RESULT] regression: {metrics}")

    json_path = args.json_output or Path(f"{args.data.stem}-FT-metrics.json")
    csv_path = args.csv_output or Path(f"{args.data.stem}-FT-metrics.csv")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    columns = [
        "Model",
        "Task",
        "Split",
        "auc_int",
        "acc_int",
        "f1_int",
        "sens_int",
        "spec_int",
        "auc_ext",
        "acc_ext",
        "f1_ext",
        "sens_ext",
        "spec_ext",
        "auc_int_lower",
        "auc_int_upper",
        "auc_ext_lower",
        "auc_ext_upper",
        "acc_int_lower",
        "acc_int_upper",
        "acc_ext_lower",
        "acc_ext_upper",
        "f1_int_lower",
        "f1_int_upper",
        "f1_ext_lower",
        "f1_ext_upper",
        "sens_int_lower",
        "sens_int_upper",
        "sens_ext_lower",
        "sens_ext_upper",
        "spec_int_lower",
        "spec_int_upper",
        "spec_ext_lower",
        "spec_ext_upper",
        "r2_int",
        "mae_int",
        "rmse_int",
        "r2_ext",
        "mae_ext",
        "rmse_ext",
        "r2_int_lower",
        "r2_int_upper",
        "r2_ext_lower",
        "r2_ext_upper",
        "mae_int_lower",
        "mae_int_upper",
        "mae_ext_lower",
        "mae_ext_upper",
        "rmse_int_lower",
        "rmse_int_upper",
        "rmse_ext_lower",
        "rmse_ext_upper",
    ]
    df_out = pd.DataFrame(results)
    df_out = df_out.reindex(columns=columns)
    df_out.to_csv(csv_path, index=False)
    print(f"[INFO] Saved metrics to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
