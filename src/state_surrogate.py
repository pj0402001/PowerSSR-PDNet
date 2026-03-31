"""
Full-state surrogate model for case9mod.

Goal:
- Input: generator set-points (P_G2, P_G3)
- Output:
  1) security feasibility probability
  2) system state variables (P_G1, Q_G1..Q_G3, V1..V9, theta1..theta9)

This module is designed for replacing expensive point-wise traditional scans
with one-shot neural inference while still exposing internal physical states.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


CASE9_STATE_COLUMNS: List[str] = [
    "p1_mw",
    "q1_mvar",
    "q2_mvar",
    "q3_mvar",
    "v1_pu",
    "v2_pu",
    "v3_pu",
    "v4_pu",
    "v5_pu",
    "v6_pu",
    "v7_pu",
    "v8_pu",
    "v9_pu",
    "theta1_deg",
    "theta2_deg",
    "theta3_deg",
    "theta4_deg",
    "theta5_deg",
    "theta6_deg",
    "theta7_deg",
    "theta8_deg",
    "theta9_deg",
]


def _dominant_step(values: np.ndarray, fallback: float) -> float:
    vals = np.unique(np.round(values.astype(np.float64), 6))
    if len(vals) < 2:
        return fallback
    diffs = np.diff(vals)
    diffs = diffs[diffs > 1e-9]
    if len(diffs) == 0:
        return fallback
    uniq, cnt = np.unique(np.round(diffs, 6), return_counts=True)
    step = float(uniq[np.argmax(cnt)])
    return step if step > 0 else fallback


def _safe_std(x: np.ndarray, axis: int = 0) -> np.ndarray:
    s = x.std(axis=axis)
    s = np.where(s < 1e-6, 1.0, s)
    return s


def _neighbor_count_8(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.int16)
    p = np.pad(m, 1, mode="constant", constant_values=0)
    return (
        p[:-2, :-2]
        + p[:-2, 1:-1]
        + p[:-2, 2:]
        + p[1:-1, :-2]
        + p[1:-1, 2:]
        + p[2:, :-2]
        + p[2:, 1:-1]
        + p[2:, 2:]
    )


@dataclass
class Case9Dataset:
    X_raw: np.ndarray
    X_norm: np.ndarray
    y_cls: np.ndarray
    y_state_raw: np.ndarray
    y_state_norm: np.ndarray
    state_mask: np.ndarray
    boundary_mask: np.ndarray
    state_names: List[str]
    x_mean: np.ndarray
    x_std: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray
    p2_grid: np.ndarray
    p3_grid: np.ndarray
    total_load_mw: float
    secure_lookup: Dict[Tuple[int, int], np.ndarray]
    secure_points: np.ndarray
    insecure_points: np.ndarray


def load_case9mod_traditional(data_dir: Path) -> pd.DataFrame:
    csv_path = data_dir / "ac_opf_9results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Traditional case9mod CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = ["p2_mw", "p3_mw"] + CASE9_STATE_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    return df


def build_case9mod_dataset(
    data_dir: Path,
    seed: int = 42,
    bg_multiplier: float = 6.0,
    n_guard_max: int = 3000,
) -> Case9Dataset:
    """
    Build multitask dataset:
    - feasible samples from traditional OPF table with full state targets
    - infeasible samples from lattice complement + domain guard negatives
    """
    rng = np.random.default_rng(seed)
    df = load_case9mod_traditional(data_dir)
    total_load_mw = float(df["total_load"].iloc[0]) if "total_load" in df.columns else 189.0

    # Canonical scan lattice from original pipeline
    p2_grid = np.linspace(10.0, 300.0, 300, dtype=np.float32)
    p3_grid = np.linspace(10.0, 270.0, 300, dtype=np.float32)
    d2 = float(p2_grid[1] - p2_grid[0])
    d3 = float(p3_grid[1] - p3_grid[0])

    # Snap feasible points to lattice keys and deduplicate
    ix = np.rint((df["p2_mw"].to_numpy(dtype=np.float64) - p2_grid[0]) / d2).astype(int)
    iy = np.rint((df["p3_mw"].to_numpy(dtype=np.float64) - p3_grid[0]) / d3).astype(int)
    valid = (ix >= 0) & (ix < len(p2_grid)) & (iy >= 0) & (iy < len(p3_grid))

    df = df.loc[valid].copy().reset_index(drop=True)
    ix = ix[valid]
    iy = iy[valid]
    df["_ix"] = ix
    df["_iy"] = iy
    df = df.drop_duplicates(subset=["_ix", "_iy"], keep="first").reset_index(drop=True)

    ix = df["_ix"].to_numpy(dtype=int)
    iy = df["_iy"].to_numpy(dtype=int)

    secure_mask_2d = np.zeros((len(p3_grid), len(p2_grid)), dtype=bool)
    secure_mask_2d[iy, ix] = True
    secure_neighbor_cnt = _neighbor_count_8(secure_mask_2d)
    insecure_neighbor_cnt = _neighbor_count_8(~secure_mask_2d)
    boundary_secure_mask = secure_mask_2d & (insecure_neighbor_cnt > 0)
    boundary_insecure_mask = (~secure_mask_2d) & (secure_neighbor_cnt > 0)

    # Feasible set with state labels
    X_feas = np.column_stack([p2_grid[ix], p3_grid[iy]]).astype(np.float32)
    Y_feas_state = df[CASE9_STATE_COLUMNS].to_numpy(dtype=np.float32)
    y_feas = np.ones(len(X_feas), dtype=np.float32)
    boundary_feas = boundary_secure_mask[iy, ix].astype(np.float32)

    secure_lookup: Dict[Tuple[int, int], np.ndarray] = {
        (int(i), int(j)): state.astype(np.float32)
        for i, j, state in zip(ix, iy, Y_feas_state)
    }

    # Infeasible from lattice complement (boundary-prioritized sampling)
    insecure_idx = np.argwhere(~secure_mask_2d)
    boundary_insecure_idx = np.argwhere(boundary_insecure_mask)
    interior_insecure_idx = np.argwhere((~secure_mask_2d) & (~boundary_insecure_mask))

    n_bg = min(int(len(X_feas) * bg_multiplier), len(insecure_idx))
    n_bg_boundary = min(len(boundary_insecure_idx), n_bg)

    picked_parts: List[np.ndarray] = []
    picked_boundary_flags: List[np.ndarray] = []
    if n_bg_boundary > 0:
        if n_bg_boundary == len(boundary_insecure_idx):
            pick_bd = boundary_insecure_idx
        else:
            take_bd = rng.choice(len(boundary_insecure_idx), size=n_bg_boundary, replace=False)
            pick_bd = boundary_insecure_idx[take_bd]
        picked_parts.append(pick_bd)
        picked_boundary_flags.append(np.ones(len(pick_bd), dtype=np.float32))

    n_rest = n_bg - n_bg_boundary
    if n_rest > 0 and len(interior_insecure_idx) > 0:
        n_take = min(n_rest, len(interior_insecure_idx))
        take_in = rng.choice(len(interior_insecure_idx), size=n_take, replace=False)
        pick_in = interior_insecure_idx[take_in]
        picked_parts.append(pick_in)
        picked_boundary_flags.append(np.zeros(len(pick_in), dtype=np.float32))

    if picked_parts:
        pick = np.vstack(picked_parts).astype(int)
        boundary_bg = np.concatenate(picked_boundary_flags).astype(np.float32)
    else:
        pick = np.zeros((0, 2), dtype=int)
        boundary_bg = np.zeros((0,), dtype=np.float32)

    X_bg = np.column_stack([p2_grid[pick[:, 1]], p3_grid[pick[:, 0]]]).astype(np.float32)

    # Domain-guard negatives below lower active-power bounds
    n_guard = min(len(X_feas), int(n_guard_max))
    n_guard_x = n_guard // 2
    n_guard_y = n_guard - n_guard_x

    x_floor = float(p2_grid[0])
    y_floor = float(p3_grid[0])
    vis_hi = 180.0
    X_guard_x = np.column_stack(
        [rng.uniform(0.0, x_floor, size=n_guard_x), rng.uniform(0.0, vis_hi, size=n_guard_x)]
    ).astype(np.float32)
    X_guard_y = np.column_stack(
        [rng.uniform(0.0, vis_hi, size=n_guard_y), rng.uniform(0.0, y_floor, size=n_guard_y)]
    ).astype(np.float32)
    X_guard = np.vstack([X_guard_x, X_guard_y]).astype(np.float32)

    X_infeas = np.vstack([X_bg, X_guard]).astype(np.float32)
    y_infeas = np.zeros(len(X_infeas), dtype=np.float32)
    boundary_infeas = np.concatenate(
        [
            boundary_bg,
            np.zeros(len(X_guard), dtype=np.float32),
        ]
    ).astype(np.float32)

    # Assemble multitask arrays
    X_raw = np.vstack([X_feas, X_infeas]).astype(np.float32)
    y_cls = np.concatenate([y_feas, y_infeas]).astype(np.float32)
    boundary_mask = np.concatenate([boundary_feas, boundary_infeas]).astype(np.float32)

    n_state = len(CASE9_STATE_COLUMNS)
    y_state_raw = np.zeros((len(X_raw), n_state), dtype=np.float32)
    state_mask = np.zeros(len(X_raw), dtype=np.float32)
    y_state_raw[: len(X_feas)] = Y_feas_state
    state_mask[: len(X_feas)] = 1.0

    x_mean = np.array([163.0, 85.0], dtype=np.float32)
    x_std = np.array([145.0, 130.0], dtype=np.float32)
    X_norm = (X_raw - x_mean) / x_std

    state_mean = y_state_raw[state_mask > 0.5].mean(axis=0).astype(np.float32)
    state_std = _safe_std(y_state_raw[state_mask > 0.5], axis=0).astype(np.float32)
    y_state_norm = np.zeros_like(y_state_raw)
    y_state_norm[state_mask > 0.5] = (
        (y_state_raw[state_mask > 0.5] - state_mean) / state_std
    )

    return Case9Dataset(
        X_raw=X_raw,
        X_norm=X_norm,
        y_cls=y_cls,
        y_state_raw=y_state_raw,
        y_state_norm=y_state_norm,
        state_mask=state_mask,
        boundary_mask=boundary_mask,
        state_names=list(CASE9_STATE_COLUMNS),
        x_mean=x_mean,
        x_std=x_std,
        state_mean=state_mean,
        state_std=state_std,
        p2_grid=p2_grid,
        p3_grid=p3_grid,
        total_load_mw=total_load_mw,
        secure_lookup=secure_lookup,
        secure_points=X_feas,
        insecure_points=X_infeas,
    )


def split_indices(y_cls: np.ndarray, seed: int = 42) -> Dict[str, np.ndarray]:
    idx = np.arange(len(y_cls))
    idx_train, idx_tmp = train_test_split(
        idx,
        test_size=0.30,
        random_state=seed,
        stratify=y_cls,
    )
    idx_val, idx_test = train_test_split(
        idx_tmp,
        test_size=0.50,
        random_state=seed + 1,
        stratify=y_cls[idx_tmp],
    )
    out: Dict[str, np.ndarray] = {
        "train": idx_train,
        "val": idx_val,
        "test": idx_test,
    }
    return out


class MultiTaskDataset(Dataset):
    def __init__(
        self,
        X_norm: np.ndarray,
        y_cls: np.ndarray,
        y_state_norm: np.ndarray,
        state_mask: np.ndarray,
        boundary_mask: np.ndarray,
    ):
        self.X = torch.from_numpy(X_norm.astype(np.float32))
        self.y_cls = torch.from_numpy(y_cls.astype(np.float32))
        self.y_state = torch.from_numpy(y_state_norm.astype(np.float32))
        self.mask = torch.from_numpy(state_mask.astype(np.float32))
        self.boundary = torch.from_numpy(boundary_mask.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y_cls[idx], self.y_state[idx], self.mask[idx], self.boundary[idx]


def make_dataloaders(
    dataset: Case9Dataset,
    split: Dict[str, np.ndarray],
    batch_size: int = 512,
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for name, ids in split.items():
        ds = MultiTaskDataset(
            dataset.X_norm[ids],
            dataset.y_cls[ids],
            dataset.y_state_norm[ids],
            dataset.state_mask[ids],
            dataset.boundary_mask[ids],
        )
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            drop_last=False,
        )
    return loaders


class FullStatePDNet(nn.Module):
    """Shared-encoder multitask model for classification + state regression."""

    def __init__(
        self,
        input_dim: int,
        n_state: int,
        trunk_dims: Optional[Sequence[int]] = None,
        cls_dims: Optional[Sequence[int]] = None,
        state_dims: Optional[Sequence[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        trunk_dims = list(trunk_dims) if trunk_dims is not None else [384, 384, 256]
        cls_dims = list(cls_dims) if cls_dims is not None else [192, 96]
        state_dims = list(state_dims) if state_dims is not None else [256, 192, 128]

        feat_layers: List[nn.Module] = []
        prev = input_dim
        for h in trunk_dims:
            feat_layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.LayerNorm(h),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        self.encoder = nn.Sequential(*feat_layers)
        feat_dim = prev

        cls_layers: List[nn.Module] = []
        prev = feat_dim
        for h in cls_dims:
            cls_layers.extend([nn.Linear(prev, h), nn.SiLU(), nn.Dropout(dropout * 0.5)])
            prev = h
        cls_layers.append(nn.Linear(prev, 1))
        self.cls_head = nn.Sequential(*cls_layers)

        state_layers: List[nn.Module] = []
        prev = feat_dim
        for h in state_dims:
            state_layers.extend([nn.Linear(prev, h), nn.SiLU(), nn.Dropout(dropout * 0.5)])
            prev = h
        state_layers.append(nn.Linear(prev, n_state))
        self.state_head = nn.Sequential(*state_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        logit = self.cls_head(feat).squeeze(-1)
        state_norm = self.state_head(feat)
        return logit, state_norm


class EnergyClosurePDNet(nn.Module):
    r"""
    Physics-closure-aware multitask model.

    Key idea:
    - Instead of directly regressing P_G1, predict network loss \hat{P}_loss >= 0
      and recover slack generation via active-power closure:
      P_G1 = P_load + \hat{P}_loss - P_G2 - P_G3.
    - This reduces unconstrained regression freedom and enforces physically
      consistent active-power balance by construction.
    """

    def __init__(
        self,
        input_dim: int,
        n_state: int,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        total_load_mw: float,
        trunk_dims: Optional[Sequence[int]] = None,
        cls_dims: Optional[Sequence[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if n_state != len(CASE9_STATE_COLUMNS):
            raise ValueError("EnergyClosurePDNet currently expects case9mod 22-state target")

        trunk_dims = list(trunk_dims) if trunk_dims is not None else [384, 384, 256]
        cls_dims = list(cls_dims) if cls_dims is not None else [192, 96]

        feat_layers: List[nn.Module] = []
        prev = input_dim
        for h in trunk_dims:
            feat_layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.LayerNorm(h),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        self.encoder = nn.Sequential(*feat_layers)
        feat_dim = prev

        cls_layers: List[nn.Module] = []
        prev = feat_dim
        for h in cls_dims:
            cls_layers.extend([nn.Linear(prev, h), nn.SiLU(), nn.Dropout(dropout * 0.5)])
            prev = h
        cls_layers.append(nn.Linear(prev, 1))
        self.cls_head = nn.Sequential(*cls_layers)

        # Physics-structured state heads
        self.loss_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        self.q_head = nn.Sequential(
            nn.Linear(feat_dim, 192),
            nn.SiLU(),
            nn.Linear(192, 3),
        )
        self.v_head = nn.Sequential(
            nn.Linear(feat_dim, 192),
            nn.SiLU(),
            nn.Linear(192, 9),
        )
        self.theta_head = nn.Sequential(
            nn.Linear(feat_dim, 192),
            nn.SiLU(),
            nn.Linear(192, 8),
        )

        # Register constants/buffers for differentiable denormalization
        self.register_buffer("x_mean_t", torch.from_numpy(x_mean.astype(np.float32)).view(1, -1))
        self.register_buffer("x_std_t", torch.from_numpy(x_std.astype(np.float32)).view(1, -1))
        self.register_buffer("state_mean_t", torch.from_numpy(state_mean.astype(np.float32)).view(1, -1))
        self.register_buffer("state_std_t", torch.from_numpy(state_std.astype(np.float32)).view(1, -1))
        self.register_buffer("total_load_t", torch.tensor([float(total_load_mw)], dtype=torch.float32).view(1, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        logit = self.cls_head(feat).squeeze(-1)

        # Inputs in physical MW
        x_raw = x * self.x_std_t + self.x_mean_t
        p2 = x_raw[:, 0:1]
        p3 = x_raw[:, 1:2]

        # Positive loss prediction (MW)
        p_loss = F.softplus(self.loss_head(feat)) + 0.5
        p1 = self.total_load_t + p_loss - p2 - p3

        # Bound-aware outputs
        q_raw = -5.0 + (300.0 + 5.0) * torch.sigmoid(self.q_head(feat))
        v_raw = 0.9 + 0.2 * torch.sigmoid(self.v_head(feat))

        theta_rest = self.theta_head(feat)
        theta1 = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        theta_raw = torch.cat([theta1, theta_rest], dim=1)

        state_raw = torch.cat([p1, q_raw, v_raw, theta_raw], dim=1)
        state_norm = (state_raw - self.state_mean_t) / self.state_std_t
        return logit, state_norm


def build_state_weight_vector(
    state_names: Sequence[str],
    p1_weight: float = 3.0,
    q_weight: float = 1.2,
    v_weight: float = 1.0,
    theta_weight: float = 1.0,
) -> np.ndarray:
    w = np.ones(len(state_names), dtype=np.float32)
    for i, name in enumerate(state_names):
        if name == "p1_mw":
            w[i] = float(p1_weight)
        elif name.startswith("q"):
            w[i] = float(q_weight)
        elif name.startswith("v"):
            w[i] = float(v_weight)
        elif name.startswith("theta"):
            w[i] = float(theta_weight)
    return w


def _weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # SmoothL1 per output dimension with manual weighting.
    diff = F.smooth_l1_loss(pred, target, reduction="none")
    w = weights.view(1, -1)
    return (diff * w).mean()


def _compute_cls_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (probs > threshold).astype(np.int64)
    lab = labels.astype(np.int64)
    tp = int(((pred == 1) & (lab == 1)).sum())
    tn = int(((pred == 0) & (lab == 0)).sum())
    fp = int(((pred == 1) & (lab == 0)).sum())
    fn = int(((pred == 0) & (lab == 1)).sum())
    spec = tn / (tn + fp + 1e-12)
    return {
        "acc": float(accuracy_score(lab, pred)),
        "prec": float(precision_score(lab, pred, zero_division=0.0)),
        "rec": float(recall_score(lab, pred, zero_division=0.0)),
        "f1": float(f1_score(lab, pred, zero_division=0.0)),
        "spec": float(spec),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best_th = 0.5
    best_metrics = _compute_cls_metrics(probs, labels, threshold=best_th)
    best_f1 = best_metrics["f1"]
    for th in np.linspace(0.10, 0.90, 81):
        m = _compute_cls_metrics(probs, labels, threshold=float(th))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_th = float(th)
            best_metrics = m
    return best_th, best_metrics


def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    all_x_norm: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_state_pred: List[np.ndarray] = []
    all_state_true: List[np.ndarray] = []
    all_mask: List[np.ndarray] = []
    all_boundary: List[np.ndarray] = []

    with torch.no_grad():
        for xb, yb, sb, mb, bb in loader:
            xb = xb.to(device)
            logits, state_pred = model(xb)
            probs = torch.sigmoid(logits)

            all_x_norm.append(xb.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(yb.numpy())
            all_state_pred.append(state_pred.cpu().numpy())
            all_state_true.append(sb.numpy())
            all_mask.append(mb.numpy())
            all_boundary.append(bb.numpy())

    return {
        "x_norm": np.concatenate(all_x_norm, axis=0),
        "probs": np.concatenate(all_probs, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "state_pred_norm": np.concatenate(all_state_pred, axis=0),
        "state_true_norm": np.concatenate(all_state_true, axis=0),
        "mask": np.concatenate(all_mask, axis=0),
        "boundary": np.concatenate(all_boundary, axis=0),
    }


def train_full_state_model(
    model: FullStatePDNet,
    loaders: Dict[str, DataLoader],
    dataset: Case9Dataset,
    device: torch.device,
    epochs: int = 160,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_state: float = 1.0,
    lambda_voltage: float = 0.05,
    lambda_monotonic: float = 0.0,
    lambda_polar: float = 0.25,
    lambda_hard_neg: float = 0.35,
    hard_neg_th: float = 0.20,
    lambda_hard_pos: float = 0.15,
    hard_pos_floor: float = 0.80,
    boundary_weight: float = 1.5,
    boundary_hard_neg_boost: float = 2.0,
    patience: int = 30,
    state_weight: Optional[np.ndarray] = None,
) -> Dict:
    model = model.to(device)

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    n_pos = float(dataset.y_cls[dataset.y_cls > 0.5].shape[0])
    n_neg = float(dataset.y_cls[dataset.y_cls <= 0.5].shape[0])
    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], dtype=torch.float32, device=device)

    loss_cls_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    state_mean_t = torch.from_numpy(dataset.state_mean).to(device)
    state_std_t = torch.from_numpy(dataset.state_std).to(device)
    voltage_idx = [i for i, n in enumerate(dataset.state_names) if n.startswith("v") and n.endswith("_pu")]
    x_std_t = torch.from_numpy(dataset.x_std).to(device)
    if state_weight is None:
        state_w_t = torch.ones(len(dataset.state_names), dtype=torch.float32, device=device)
    else:
        state_w_t = torch.from_numpy(state_weight.astype(np.float32)).to(device)

    history = {
        "train_total": [],
        "train_cls": [],
        "train_polar": [],
        "train_hard_neg": [],
        "train_hard_pos": [],
        "train_state": [],
        "train_voltage": [],
        "train_mono": [],
        "val_total": [],
        "val_f1@0.5": [],
        "val_state_mae": [],
    }

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        ep_total: List[float] = []
        ep_cls: List[float] = []
        ep_polar: List[float] = []
        ep_hard_neg: List[float] = []
        ep_hard_pos: List[float] = []
        ep_state: List[float] = []
        ep_v: List[float] = []
        ep_mono: List[float] = []

        for xb, yb, sb, mb, bb in train_loader:
            xb = xb.to(device)
            if lambda_monotonic > 0:
                xb.requires_grad_(True)
            yb = yb.to(device)
            sb = sb.to(device)
            mb = mb.to(device)
            bb = bb.to(device)

            logits, state_pred = model(xb)
            probs = torch.sigmoid(logits)

            cls_raw = loss_cls_fn(logits, yb)
            cls_w = 1.0 + boundary_weight * bb
            loss_cls = (cls_raw * cls_w).mean()

            polar_raw = (1.0 - yb) * (probs ** 2) + yb * ((1.0 - probs) ** 2)
            loss_polar = (polar_raw * cls_w).mean()

            hard_neg = (yb < 0.5) & (probs > hard_neg_th)
            if hard_neg.any():
                hn_w = 1.0 + boundary_hard_neg_boost * bb[hard_neg]
                loss_hard_neg = (((probs[hard_neg] - hard_neg_th) ** 2) * hn_w).mean()
            else:
                loss_hard_neg = torch.tensor(0.0, device=device)

            hard_pos = (yb > 0.5) & (probs < hard_pos_floor)
            if hard_pos.any():
                loss_hard_pos = ((hard_pos_floor - probs[hard_pos]) ** 2).mean()
            else:
                loss_hard_pos = torch.tensor(0.0, device=device)

            has_state = mb > 0.5
            if has_state.any():
                loss_state = _weighted_smooth_l1(state_pred[has_state], sb[has_state], state_w_t)

                state_raw = state_pred[has_state] * state_std_t + state_mean_t
                v_raw = state_raw[:, voltage_idx]
                v_viol = F.relu(0.90 - v_raw) + F.relu(v_raw - 1.10)
                loss_voltage = v_viol.mean()
            else:
                loss_state = torch.tensor(0.0, device=device)
                loss_voltage = torch.tensor(0.0, device=device)

            loss_mono = torch.tensor(0.0, device=device)
            if lambda_monotonic > 0:
                # Monotonic prior: with fixed loads, slack P_G1 should decrease
                # as either controllable generation (P_G2/P_G3) increases.
                p1_raw_all = state_pred[:, 0] * state_std_t[0] + state_mean_t[0]
                grad_all = torch.autograd.grad(
                    p1_raw_all.sum(),
                    xb,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=False,
                )[0]
                # Convert d/dx_norm to d/dx_raw by dividing by std (>0, sign preserved)
                dp1_dp2 = grad_all[:, 0] / (x_std_t[0] + 1e-12)
                dp1_dp3 = grad_all[:, 1] / (x_std_t[1] + 1e-12)
                loss_mono = F.relu(dp1_dp2 + 0.02).mean() + F.relu(dp1_dp3 + 0.02).mean()

            loss = (
                loss_cls
                + lambda_polar * loss_polar
                + lambda_hard_neg * loss_hard_neg
                + lambda_hard_pos * loss_hard_pos
                + lambda_state * loss_state
                + lambda_voltage * loss_voltage
                + lambda_monotonic * loss_mono
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            ep_total.append(float(loss.item()))
            ep_cls.append(float(loss_cls.item()))
            ep_polar.append(float(loss_polar.item()))
            ep_hard_neg.append(float(loss_hard_neg.item()))
            ep_hard_pos.append(float(loss_hard_pos.item()))
            ep_state.append(float(loss_state.item()))
            ep_v.append(float(loss_voltage.item()))
            ep_mono.append(float(loss_mono.item()))

        scheduler.step()

        # Validation (loss + coarse metrics)
        model.eval()
        val_total: List[float] = []
        with torch.no_grad():
            for xb, yb, sb, mb, bb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                sb = sb.to(device)
                mb = mb.to(device)
                bb = bb.to(device)

                logits, state_pred = model(xb)
                probs = torch.sigmoid(logits)

                cls_raw = loss_cls_fn(logits, yb)
                cls_w = 1.0 + boundary_weight * bb
                loss_cls = (cls_raw * cls_w).mean()

                polar_raw = (1.0 - yb) * (probs ** 2) + yb * ((1.0 - probs) ** 2)
                loss_polar = (polar_raw * cls_w).mean()

                hard_neg = (yb < 0.5) & (probs > hard_neg_th)
                if hard_neg.any():
                    hn_w = 1.0 + boundary_hard_neg_boost * bb[hard_neg]
                    loss_hard_neg = (((probs[hard_neg] - hard_neg_th) ** 2) * hn_w).mean()
                else:
                    loss_hard_neg = torch.tensor(0.0, device=device)

                hard_pos = (yb > 0.5) & (probs < hard_pos_floor)
                if hard_pos.any():
                    loss_hard_pos = ((hard_pos_floor - probs[hard_pos]) ** 2).mean()
                else:
                    loss_hard_pos = torch.tensor(0.0, device=device)

                has_state = mb > 0.5
                if has_state.any():
                    loss_state = F.smooth_l1_loss(state_pred[has_state], sb[has_state])
                    state_raw = state_pred[has_state] * state_std_t + state_mean_t
                    v_raw = state_raw[:, voltage_idx]
                    v_viol = F.relu(0.90 - v_raw) + F.relu(v_raw - 1.10)
                    loss_voltage = v_viol.mean()
                else:
                    loss_state = torch.tensor(0.0, device=device)
                    loss_voltage = torch.tensor(0.0, device=device)

                loss = (
                    loss_cls
                    + lambda_polar * loss_polar
                    + lambda_hard_neg * loss_hard_neg
                    + lambda_hard_pos * loss_hard_pos
                    + lambda_state * loss_state
                    + lambda_voltage * loss_voltage
                )
                val_total.append(float(loss.item()))

        val_pred = _collect_predictions(model, val_loader, device)
        val_metrics = _compute_cls_metrics(val_pred["probs"], val_pred["labels"], threshold=0.5)

        mask = val_pred["mask"] > 0.5
        if mask.any():
            pred_raw = val_pred["state_pred_norm"][mask] * dataset.state_std + dataset.state_mean
            true_raw = val_pred["state_true_norm"][mask] * dataset.state_std + dataset.state_mean
            val_state_mae = float(np.mean(np.abs(pred_raw - true_raw)))
        else:
            val_state_mae = float("nan")

        train_total = float(np.mean(ep_total))
        train_cls = float(np.mean(ep_cls))
        train_polar = float(np.mean(ep_polar))
        train_hard_neg = float(np.mean(ep_hard_neg))
        train_hard_pos = float(np.mean(ep_hard_pos))
        train_state = float(np.mean(ep_state))
        train_v = float(np.mean(ep_v))
        train_mono = float(np.mean(ep_mono))
        val_total_mean = float(np.mean(val_total))

        history["train_total"].append(train_total)
        history["train_cls"].append(train_cls)
        history["train_polar"].append(train_polar)
        history["train_hard_neg"].append(train_hard_neg)
        history["train_hard_pos"].append(train_hard_pos)
        history["train_state"].append(train_state)
        history["train_voltage"].append(train_v)
        history["train_mono"].append(train_mono)
        history["val_total"].append(val_total_mean)
        history["val_f1@0.5"].append(float(val_metrics["f1"]))
        history["val_state_mae"].append(val_state_mae)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"train={train_total:.4f} (cls={train_cls:.4f}, polar={train_polar:.4f}, "
                f"hardneg={train_hard_neg:.4f}, hardpos={train_hard_pos:.4f}, state={train_state:.4f}) | "
                f"mono={train_mono:.4f} | val={val_total_mean:.4f} | val_f1@0.5={val_metrics['f1']:.4f} | "
                f"val_state_mae={val_state_mae:.4f}"
            )

        if val_total_mean < best_val - 1e-5:
            best_val = val_total_mean
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)

    # Final threshold from validation set
    val_pred = _collect_predictions(model, val_loader, device)
    best_th, val_best = find_best_threshold(val_pred["probs"], val_pred["labels"])
    history["best_threshold"] = float(best_th)
    history["val_best"] = val_best
    return history


def evaluate_full_state_model(
    model: FullStatePDNet,
    loader: DataLoader,
    dataset: Case9Dataset,
    device: torch.device,
    threshold: float,
) -> Dict:
    pred = _collect_predictions(model, loader, device)
    cls = _compute_cls_metrics(pred["probs"], pred["labels"], threshold=threshold)
    cls["best_threshold"] = float(threshold)

    probs = pred["probs"].astype(np.float64)
    labels = pred["labels"].astype(np.int64)
    boundary = pred.get("boundary", np.zeros_like(labels, dtype=np.float32)) > 0.5

    pos = probs[labels == 1]
    neg = probs[labels == 0]

    def _q(arr: np.ndarray, q: float) -> float:
        if arr.size == 0:
            return float("nan")
        return float(np.percentile(arr, q))

    prob_stats = {
        "pos_mean": float(pos.mean()) if pos.size else float("nan"),
        "pos_p05": _q(pos, 5.0),
        "pos_p50": _q(pos, 50.0),
        "pos_p95": _q(pos, 95.0),
        "pos_lt_0p5": float((pos < 0.5).mean()) if pos.size else float("nan"),
        "neg_mean": float(neg.mean()) if neg.size else float("nan"),
        "neg_p50": _q(neg, 50.0),
        "neg_p95": _q(neg, 95.0),
        "neg_p99": _q(neg, 99.0),
        "neg_gt_0p2": float((neg > 0.2).mean()) if neg.size else float("nan"),
        "neg_gt_0p5": float((neg > 0.5).mean()) if neg.size else float("nan"),
    }

    neg_boundary = probs[(labels == 0) & boundary]
    neg_interior = probs[(labels == 0) & (~boundary)]
    prob_stats["neg_boundary_n"] = int(neg_boundary.size)
    prob_stats["neg_boundary_mean"] = float(neg_boundary.mean()) if neg_boundary.size else float("nan")
    prob_stats["neg_boundary_p95"] = _q(neg_boundary, 95.0)
    prob_stats["neg_boundary_gt_0p5"] = float((neg_boundary > 0.5).mean()) if neg_boundary.size else float("nan")
    prob_stats["neg_interior_n"] = int(neg_interior.size)
    prob_stats["neg_interior_mean"] = float(neg_interior.mean()) if neg_interior.size else float("nan")
    prob_stats["neg_interior_p95"] = _q(neg_interior, 95.0)
    prob_stats["neg_interior_gt_0p5"] = float((neg_interior > 0.5).mean()) if neg_interior.size else float("nan")

    mask = pred["mask"] > 0.5
    if mask.any():
        state_pred_raw = pred["state_pred_norm"][mask] * dataset.state_std + dataset.state_mean
        state_true_raw = pred["state_true_norm"][mask] * dataset.state_std + dataset.state_mean

        abs_err = np.abs(state_pred_raw - state_true_raw)
        sq_err = (state_pred_raw - state_true_raw) ** 2

        mae_by_var = {
            name: float(val)
            for name, val in zip(dataset.state_names, abs_err.mean(axis=0))
        }
        rmse_by_var = {
            name: float(np.sqrt(val))
            for name, val in zip(dataset.state_names, sq_err.mean(axis=0))
        }

        def _group_mean(prefix: str) -> float:
            idx = [i for i, n in enumerate(dataset.state_names) if n.startswith(prefix)]
            if not idx:
                return float("nan")
            return float(abs_err[:, idx].mean())

        state_metrics = {
            "n_feasible_eval": int(mask.sum()),
            "overall_mae": float(abs_err.mean()),
            "overall_rmse": float(np.sqrt(sq_err.mean())),
            "mae_group": {
                "p_slack": float(abs_err[:, [0]].mean()),
                "q": _group_mean("q"),
                "v": _group_mean("v"),
                "theta": _group_mean("theta"),
            },
            "mae_by_var": mae_by_var,
            "rmse_by_var": rmse_by_var,
        }
    else:
        state_metrics = {
            "n_feasible_eval": 0,
            "overall_mae": float("nan"),
            "overall_rmse": float("nan"),
            "mae_group": {},
            "mae_by_var": {},
            "rmse_by_var": {},
        }

    return {
        "classification": cls,
        "probability": prob_stats,
        "state": state_metrics,
    }


def evaluate_energy_consistency(
    model: nn.Module,
    loader: DataLoader,
    dataset: Case9Dataset,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate active-power closure residual:
    r_P = P_G1 + P_G2 + P_G3 - P_load - P_loss_true.

    Here P_loss_true is estimated from traditional feasible states as
    (P_G1 + P_G2 + P_G3 - P_load). For infeasible points, this metric is skipped.
    """
    pred = _collect_predictions(model, loader, device)
    mask = pred["mask"] > 0.5
    if not mask.any():
        return {
            "n_feasible_eval": 0,
            "closure_abs_mean_mw": float("nan"),
            "closure_abs_p95_mw": float("nan"),
        }

    x_raw = pred["x_norm"] * dataset.x_std + dataset.x_mean
    state_pred_raw = pred["state_pred_norm"] * dataset.state_std + dataset.state_mean
    state_true_raw = pred["state_true_norm"] * dataset.state_std + dataset.state_mean

    p2 = x_raw[mask, 0]
    p3 = x_raw[mask, 1]
    p1_pred = state_pred_raw[mask, 0]
    p1_true = state_true_raw[mask, 0]

    p_loss_true = p1_true + p2 + p3 - dataset.total_load_mw
    closure = p1_pred + p2 + p3 - dataset.total_load_mw - p_loss_true
    abs_closure = np.abs(closure)

    return {
        "n_feasible_eval": int(mask.sum()),
        "closure_abs_mean_mw": float(abs_closure.mean()),
        "closure_abs_p95_mw": float(np.percentile(abs_closure, 95.0)),
    }


def compare_points(
    model: FullStatePDNet,
    points_raw: np.ndarray,
    dataset: Case9Dataset,
    device: torch.device,
    threshold: float,
) -> List[Dict]:
    model.eval()
    p2_min = float(dataset.p2_grid[0])
    p3_min = float(dataset.p3_grid[0])
    d2 = _dominant_step(dataset.p2_grid, fallback=float(dataset.p2_grid[1] - dataset.p2_grid[0]))
    d3 = _dominant_step(dataset.p3_grid, fallback=float(dataset.p3_grid[1] - dataset.p3_grid[0]))

    out: List[Dict] = []
    for p2, p3 in points_raw:
        x = np.array([[p2, p3]], dtype=np.float32)
        x_norm = (x - dataset.x_mean) / dataset.x_std
        xb = torch.from_numpy(x_norm).to(device)

        with torch.no_grad():
            logit, state_pred_norm = model(xb)
            prob = float(torch.sigmoid(logit).item())
            state_pred = (
                state_pred_norm.cpu().numpy()[0] * dataset.state_std + dataset.state_mean
            )

        ix = int(np.rint((float(p2) - p2_min) / d2))
        iy = int(np.rint((float(p3) - p3_min) / d3))
        in_grid = (0 <= ix < len(dataset.p2_grid)) and (0 <= iy < len(dataset.p3_grid))
        key = (ix, iy)
        trad_secure = int(in_grid and (key in dataset.secure_lookup))

        row: Dict = {
            "input": {"p2_mw": float(p2), "p3_mw": float(p3)},
            "grid_key": {"ix": int(ix), "iy": int(iy)},
            "traditional_secure": trad_secure,
            "model_prob_secure": prob,
            "threshold": float(threshold),
            "model_secure": int(prob > threshold),
            "agreement": int(int(prob > threshold) == trad_secure),
        }

        if trad_secure:
            state_true = dataset.secure_lookup[key]
            abs_err = np.abs(state_pred - state_true)
            row["state_mae_overall"] = float(abs_err.mean())
            row["state_mae_group"] = {
                "p_slack": float(abs_err[[0]].mean()),
                "q": float(abs_err[1:4].mean()),
                "v": float(abs_err[4:13].mean()),
                "theta": float(abs_err[13:22].mean()),
            }
            row["state_pred"] = {
                name: float(val)
                for name, val in zip(dataset.state_names, state_pred)
            }
            row["state_true"] = {
                name: float(val)
                for name, val in zip(dataset.state_names, state_true)
            }
        else:
            row["state_mae_overall"] = None
            row["state_mae_group"] = None

        out.append(row)
    return out


def sample_demo_points(dataset: Case9Dataset, seed: int = 42, n_each: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    secure = dataset.secure_points
    insecure = dataset.insecure_points

    n_s = min(n_each, len(secure))
    n_i = min(n_each, len(insecure))

    pts_s = secure[rng.choice(len(secure), size=n_s, replace=False)]
    pts_i = insecure[rng.choice(len(insecure), size=n_i, replace=False)]
    return np.vstack([pts_s, pts_i]).astype(np.float32)


def export_checkpoint(
    model: FullStatePDNet,
    dataset: Case9Dataset,
    path: Path,
):
    payload = {
        "state_dict": model.state_dict(),
        "x_mean": dataset.x_mean,
        "x_std": dataset.x_std,
        "state_mean": dataset.state_mean,
        "state_std": dataset.state_std,
        "state_names": dataset.state_names,
    }
    torch.save(payload, path)
