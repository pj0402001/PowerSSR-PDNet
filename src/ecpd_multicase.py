"""
EC-PDNet multi-case full-state surrogate utilities.

This module extends the physics-structured full-state surrogate from case9mod
to all currently used benchmark systems in this repository:
  - WB2
  - WB5
  - case9mod
  - LMBM3 (lambda=1.490 and lambda=1.500)
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
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader, Dataset

from bukhsh_data import generate_WB2_data


def _safe_std(x: np.ndarray, axis: int = 0) -> np.ndarray:
    s = x.std(axis=axis)
    s = np.where(s < 1e-6, 1.0, s)
    return s


def _weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    diff = F.smooth_l1_loss(pred, target, reduction="none")
    return (diff * weights.view(1, -1)).mean()


def _split_indices(y_cls: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    idx = np.arange(len(y_cls))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=seed, stratify=y_cls)
    va, te = train_test_split(tmp, test_size=0.50, random_state=seed + 1, stratify=y_cls[tmp])
    return {"train": tr, "val": va, "test": te}


def split_indices(y_cls: np.ndarray, seed: int = 42) -> Dict[str, np.ndarray]:
    return _split_indices(y_cls=y_cls, seed=seed)


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


def _best_threshold(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best_th = 0.5
    best = _compute_cls_metrics(probs, labels, threshold=best_th)
    for th in np.linspace(0.1, 0.9, 81):
        m = _compute_cls_metrics(probs, labels, threshold=float(th))
        if m["f1"] > best["f1"]:
            best_th = float(th)
            best = m
    return best_th, best


@dataclass
class ClosureConfig:
    target_name: Optional[str]
    loss_name: Optional[str]
    constant_mw: float
    input_terms: List[Tuple[int, float]]
    monotonic_terms: List[Tuple[int, int]]


@dataclass
class MultiCaseDataset:
    case_id: str
    input_names: List[str]
    state_names: List[str]
    X_raw: np.ndarray
    X_norm: np.ndarray
    y_cls: np.ndarray
    y_state_raw: np.ndarray
    y_state_norm: np.ndarray
    state_mask: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray
    state_min: np.ndarray
    state_max: np.ndarray
    closure: ClosureConfig


class _TensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y_cls: np.ndarray, y_state_norm: np.ndarray, mask: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y_cls = torch.from_numpy(y_cls.astype(np.float32))
        self.y_state = torch.from_numpy(y_state_norm.astype(np.float32))
        self.mask = torch.from_numpy(mask.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y_cls[idx], self.y_state[idx], self.mask[idx]


def make_loaders(ds: MultiCaseDataset, split: Dict[str, np.ndarray], batch_size: int) -> Dict[str, DataLoader]:
    out: Dict[str, DataLoader] = {}
    for name, ids in split.items():
        td = _TensorDataset(ds.X_norm[ids], ds.y_cls[ids], ds.y_state_norm[ids], ds.state_mask[ids])
        out[name] = DataLoader(td, batch_size=batch_size, shuffle=(name == "train"), drop_last=False)
    return out


class ECPDNet(nn.Module):
    """Energy-closure PDNet for mixed case configurations."""

    def __init__(
        self,
        dataset: MultiCaseDataset,
        trunk_dims: Optional[Sequence[int]] = None,
        cls_dims: Optional[Sequence[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dataset = dataset
        self.n_state = len(dataset.state_names)

        trunk_dims = list(trunk_dims) if trunk_dims is not None else [256, 256, 128]
        cls_dims = list(cls_dims) if cls_dims is not None else [128, 64]

        feat_layers: List[nn.Module] = []
        prev = len(dataset.input_names)
        for h in trunk_dims:
            feat_layers.extend([
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
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

        self.loss_head = nn.Sequential(nn.Linear(feat_dim, 64), nn.SiLU(), nn.Linear(64, 1))

        self.target_idx = (
            dataset.state_names.index(dataset.closure.target_name)
            if dataset.closure.target_name in dataset.state_names
            else None
        )
        self.loss_idx = (
            dataset.state_names.index(dataset.closure.loss_name)
            if dataset.closure.loss_name in dataset.state_names
            else None
        )

        reserved = set(i for i in [self.target_idx, self.loss_idx] if i is not None)
        self.direct_indices = [i for i in range(self.n_state) if i not in reserved]

        self.direct_head = nn.Sequential(
            nn.Linear(feat_dim, 192),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(192, len(self.direct_indices)),
        )

        # Buffers for differentiable scaling
        self.register_buffer("x_mean_t", torch.from_numpy(dataset.x_mean.astype(np.float32)).view(1, -1))
        self.register_buffer("x_std_t", torch.from_numpy(dataset.x_std.astype(np.float32)).view(1, -1))
        self.register_buffer("state_mean_t", torch.from_numpy(dataset.state_mean.astype(np.float32)).view(1, -1))
        self.register_buffer("state_std_t", torch.from_numpy(dataset.state_std.astype(np.float32)).view(1, -1))
        self.register_buffer("state_min_t", torch.from_numpy(dataset.state_min.astype(np.float32)).view(1, -1))
        self.register_buffer("state_max_t", torch.from_numpy(dataset.state_max.astype(np.float32)).view(1, -1))
        self.register_buffer("closure_const_t", torch.tensor([dataset.closure.constant_mw], dtype=torch.float32).view(1, 1))

        # Direct-output min/max per direct index
        if len(self.direct_indices) > 0:
            dmin = dataset.state_min[self.direct_indices]
            dmax = dataset.state_max[self.direct_indices]
        else:
            dmin = np.zeros((0,), dtype=np.float32)
            dmax = np.zeros((0,), dtype=np.float32)
        self.register_buffer("direct_min_t", torch.from_numpy(dmin.astype(np.float32)).view(1, -1))
        self.register_buffer("direct_max_t", torch.from_numpy(dmax.astype(np.float32)).view(1, -1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x_norm)
        logit = self.cls_head(feat).squeeze(-1)

        x_raw = x_norm * self.x_std_t + self.x_mean_t
        p_loss = F.softplus(self.loss_head(feat)).squeeze(-1) + 1e-4

        state_raw = torch.zeros((x_norm.shape[0], self.n_state), dtype=x_norm.dtype, device=x_norm.device)

        # Fill direct outputs with range-aware mapping
        if len(self.direct_indices) > 0:
            d_raw_u = self.direct_head(feat)
            d_raw = self.direct_min_t + (self.direct_max_t - self.direct_min_t) * torch.sigmoid(d_raw_u)
            for j, idx in enumerate(self.direct_indices):
                state_raw[:, idx] = d_raw[:, j]

        # Optional explicit loss channel
        if self.loss_idx is not None:
            state_raw[:, self.loss_idx] = p_loss

        # Optional closure target
        if self.target_idx is not None:
            closure_val = self.closure_const_t.squeeze(-1) + p_loss
            for in_idx, coef in self.dataset.closure.input_terms:
                closure_val = closure_val + float(coef) * x_raw[:, in_idx]
            state_raw[:, self.target_idx] = closure_val

        return logit, state_raw


def _collect(model: ECPDNet, loader: DataLoader, device: torch.device, ds: MultiCaseDataset) -> Dict[str, np.ndarray]:
    model.eval()
    out = {"x_norm": [], "probs": [], "labels": [], "state_pred_raw": [], "state_true_norm": [], "mask": []}
    with torch.no_grad():
        for xb, yb, sb, mb in loader:
            xb = xb.to(device)
            logit, state_raw = model(xb)
            out["x_norm"].append(xb.cpu().numpy())
            out["probs"].append(torch.sigmoid(logit).cpu().numpy())
            out["labels"].append(yb.numpy())
            out["state_pred_raw"].append(state_raw.cpu().numpy())
            out["state_true_norm"].append(sb.numpy())
            out["mask"].append(mb.numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def train_ecpd(
    model: ECPDNet,
    loaders: Dict[str, DataLoader],
    ds: MultiCaseDataset,
    device: torch.device,
    epochs: int,
    lr: float,
    lambda_state: float,
    lambda_mono: float,
    patience: int,
    state_weights: Optional[np.ndarray] = None,
) -> Dict:
    model = model.to(device)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    n_pos = float(ds.y_cls[ds.y_cls > 0.5].shape[0])
    n_neg = float(ds.y_cls[ds.y_cls <= 0.5].shape[0])
    pos_w = torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], dtype=torch.float32, device=device)
    cls_loss = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    if state_weights is None:
        sw = torch.ones(len(ds.state_names), dtype=torch.float32, device=device)
    else:
        sw = torch.from_numpy(state_weights.astype(np.float32)).to(device)

    state_mean_t = torch.from_numpy(ds.state_mean.astype(np.float32)).to(device)
    state_std_t = torch.from_numpy(ds.state_std.astype(np.float32)).to(device)
    x_std_t = torch.from_numpy(ds.x_std.astype(np.float32)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

    hist: Dict[str, List[float]] = {
        "train_total": [],
        "train_cls": [],
        "train_state": [],
        "train_mono": [],
        "val_total": [],
        "val_f1@0.5": [],
        "val_state_mae": [],
    }

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    bad = 0

    for ep in range(epochs):
        model.train()
        tr_total: List[float] = []
        tr_cls: List[float] = []
        tr_state: List[float] = []
        tr_mono: List[float] = []

        for xb, yb, sb, mb in train_loader:
            xb = xb.to(device)
            if lambda_mono > 0 and model.target_idx is not None and len(ds.closure.monotonic_terms) > 0:
                xb.requires_grad_(True)
            yb = yb.to(device)
            sb = sb.to(device)
            mb = mb.to(device)

            logit, state_raw = model(xb)
            loss_c = cls_loss(logit, yb)

            state_pred_norm = (state_raw - state_mean_t.view(1, -1)) / state_std_t.view(1, -1)

            has_state = mb > 0.5
            if has_state.any():
                loss_s = _weighted_smooth_l1(state_pred_norm[has_state], sb[has_state], sw)
            else:
                loss_s = torch.tensor(0.0, device=device)

            loss_m = torch.tensor(0.0, device=device)
            if lambda_mono > 0 and model.target_idx is not None and len(ds.closure.monotonic_terms) > 0:
                target = state_raw[:, model.target_idx]
                grad = torch.autograd.grad(target.sum(), xb, create_graph=True, retain_graph=True)[0]
                penalties: List[torch.Tensor] = []
                for in_idx, sign in ds.closure.monotonic_terms:
                    d_raw = grad[:, in_idx] / (x_std_t[in_idx] + 1e-12)
                    if sign < 0:
                        penalties.append(F.relu(d_raw + 0.01))
                    else:
                        penalties.append(F.relu(0.01 - d_raw))
                if penalties:
                    loss_m = torch.stack([p.mean() for p in penalties]).mean()

            loss = loss_c + lambda_state * loss_s + lambda_mono * loss_m

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()

            tr_total.append(float(loss.item()))
            tr_cls.append(float(loss_c.item()))
            tr_state.append(float(loss_s.item()))
            tr_mono.append(float(loss_m.item()))

        sch.step()

        # Validation objective
        model.eval()
        va_loss: List[float] = []
        with torch.no_grad():
            for xb, yb, sb, mb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                sb = sb.to(device)
                mb = mb.to(device)
                logit, state_raw = model(xb)
                loss_c = cls_loss(logit, yb)
                state_pred_norm = (state_raw - state_mean_t.view(1, -1)) / state_std_t.view(1, -1)
                has_state = mb > 0.5
                if has_state.any():
                    loss_s = _weighted_smooth_l1(state_pred_norm[has_state], sb[has_state], sw)
                else:
                    loss_s = torch.tensor(0.0, device=device)
                va_loss.append(float((loss_c + lambda_state * loss_s).item()))

        pred = _collect(model, val_loader, device, ds)
        cls_m = _compute_cls_metrics(pred["probs"], pred["labels"], threshold=0.5)
        mask = pred["mask"] > 0.5
        if mask.any():
            state_true_raw = pred["state_true_norm"][mask] * ds.state_std + ds.state_mean
            state_pred_raw = pred["state_pred_raw"][mask]
            va_mae = float(np.mean(np.abs(state_pred_raw - state_true_raw)))
        else:
            va_mae = float("nan")

        hist["train_total"].append(float(np.mean(tr_total)))
        hist["train_cls"].append(float(np.mean(tr_cls)))
        hist["train_state"].append(float(np.mean(tr_state)))
        hist["train_mono"].append(float(np.mean(tr_mono)))
        hist["val_total"].append(float(np.mean(va_loss)))
        hist["val_f1@0.5"].append(float(cls_m["f1"]))
        hist["val_state_mae"].append(va_mae)

        if (ep + 1) % 10 == 0:
            print(
                f"Epoch {ep+1:3d}/{epochs} | train={hist['train_total'][-1]:.4f} "
                f"(cls={hist['train_cls'][-1]:.4f}, state={hist['train_state'][-1]:.4f}, mono={hist['train_mono'][-1]:.4f}) | "
                f"val={hist['val_total'][-1]:.4f} | val_f1@0.5={hist['val_f1@0.5'][-1]:.4f} | "
                f"val_state_mae={va_mae:.4f}"
            )

        if hist["val_total"][-1] < best_val - 1e-5:
            best_val = hist["val_total"][-1]
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {ep+1}")
                break

    model.load_state_dict(best_state)

    val_pred = _collect(model, val_loader, device, ds)
    best_th, best_val_cls = _best_threshold(val_pred["probs"], val_pred["labels"])
    hist["best_threshold"] = float(best_th)
    hist["val_best"] = best_val_cls
    return hist


def evaluate_ecpd(model: ECPDNet, loader: DataLoader, ds: MultiCaseDataset, device: torch.device, threshold: float) -> Dict:
    pred = _collect(model, loader, device, ds)
    cls = _compute_cls_metrics(pred["probs"], pred["labels"], threshold=threshold)
    cls["best_threshold"] = float(threshold)

    mask = pred["mask"] > 0.5
    if mask.any():
        state_true_raw = pred["state_true_norm"][mask] * ds.state_std + ds.state_mean
        state_pred_raw = pred["state_pred_raw"][mask]
        abs_err = np.abs(state_pred_raw - state_true_raw)
        sq_err = (state_pred_raw - state_true_raw) ** 2

        mae_by_var = {n: float(v) for n, v in zip(ds.state_names, abs_err.mean(axis=0))}
        rmse_by_var = {n: float(np.sqrt(v)) for n, v in zip(ds.state_names, sq_err.mean(axis=0))}

        groups: Dict[str, List[int]] = {"p": [], "q": [], "v": [], "theta": [], "loss": []}
        for i, n in enumerate(ds.state_names):
            ln = n.lower()
            if "loss" in ln:
                groups["loss"].append(i)
            elif ln.startswith("q"):
                groups["q"].append(i)
            elif ln.startswith("v"):
                groups["v"].append(i)
            elif "theta" in ln or ln.startswith("va"):
                groups["theta"].append(i)
            elif ln.startswith("p"):
                groups["p"].append(i)

        mae_group = {}
        for g, idx in groups.items():
            mae_group[g] = float(abs_err[:, idx].mean()) if idx else float("nan")

        state = {
            "n_state_eval": int(mask.sum()),
            "overall_mae": float(abs_err.mean()),
            "overall_rmse": float(np.sqrt(sq_err.mean())),
            "mae_group": mae_group,
            "mae_by_var": mae_by_var,
            "rmse_by_var": rmse_by_var,
        }
    else:
        state = {
            "n_state_eval": 0,
            "overall_mae": float("nan"),
            "overall_rmse": float("nan"),
            "mae_group": {},
            "mae_by_var": {},
            "rmse_by_var": {},
        }

    # Closure consistency of model prediction (should be near-zero if active)
    if model.target_idx is not None and model.loss_idx is not None:
        x_raw = pred["x_norm"] * ds.x_std + ds.x_mean
        target_pred = pred["state_pred_raw"][:, model.target_idx]
        loss_pred = pred["state_pred_raw"][:, model.loss_idx]
        rhs = ds.closure.constant_mw + loss_pred
        for in_idx, coef in ds.closure.input_terms:
            rhs = rhs + float(coef) * x_raw[:, in_idx]
        clos = np.abs(target_pred - rhs)
        closure = {
            "abs_mean": float(clos.mean()),
            "abs_p95": float(np.percentile(clos, 95.0)),
        }
    else:
        closure = {"abs_mean": float("nan"), "abs_p95": float("nan")}

    return {"classification": cls, "state": state, "closure": closure}


def sample_point_rows(
    model: ECPDNet,
    ds: MultiCaseDataset,
    split_indices: np.ndarray,
    device: torch.device,
    threshold: float,
    seed: int,
    n_each: int = 3,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    ids = split_indices
    y = ds.y_cls[ids]
    feas_ids = ids[y > 0.5]
    infeas_ids = ids[y <= 0.5]
    n_f = min(n_each, len(feas_ids))
    n_i = min(n_each, len(infeas_ids))
    pick = np.concatenate([
        rng.choice(feas_ids, size=n_f, replace=False) if n_f > 0 else np.array([], dtype=int),
        rng.choice(infeas_ids, size=n_i, replace=False) if n_i > 0 else np.array([], dtype=int),
    ])

    rows: List[Dict] = []
    model.eval()
    for idx in pick:
        x_raw = ds.X_raw[idx]
        xb = torch.from_numpy(ds.X_norm[idx:idx + 1]).to(device)
        with torch.no_grad():
            logit, state_raw = model(xb)
            prob = float(torch.sigmoid(logit).item())
            state_pred = state_raw.cpu().numpy()[0]

        row = {
            "case": ds.case_id,
            "input": {k: float(v) for k, v in zip(ds.input_names, x_raw)},
            "traditional_secure": int(ds.y_cls[idx] > 0.5),
            "model_prob_secure": prob,
            "threshold": float(threshold),
            "model_secure": int(prob > threshold),
            "agreement": int((prob > threshold) == (ds.y_cls[idx] > 0.5)),
        }

        if ds.state_mask[idx] > 0.5:
            true_state = ds.y_state_raw[idx]
            abs_err = np.abs(state_pred - true_state)
            row["state_mae_overall"] = float(abs_err.mean())
            row["state_pred"] = {k: float(v) for k, v in zip(ds.state_names, state_pred)}
            row["state_true"] = {k: float(v) for k, v in zip(ds.state_names, true_state)}
        else:
            row["state_mae_overall"] = None
            row["state_pred"] = {k: float(v) for k, v in zip(ds.state_names, state_pred)}
            row["state_true"] = None

        rows.append(row)
    return rows


def state_weight_vector(state_names: Sequence[str], p_weight: float = 3.0) -> np.ndarray:
    w = np.ones(len(state_names), dtype=np.float32)
    for i, n in enumerate(state_names):
        ln = n.lower()
        if ln.startswith("p"):
            w[i] = float(p_weight)
    return w


def _finalize_dataset(
    case_id: str,
    input_names: List[str],
    state_names: List[str],
    X_raw: np.ndarray,
    y_cls: np.ndarray,
    y_state_raw: np.ndarray,
    state_mask: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    closure: ClosureConfig,
) -> MultiCaseDataset:
    X_raw = X_raw.astype(np.float32)
    y_cls = y_cls.astype(np.float32)
    y_state_raw = y_state_raw.astype(np.float32)
    state_mask = state_mask.astype(np.float32)

    X_norm = (X_raw - x_mean.astype(np.float32)) / x_std.astype(np.float32)

    feas_state = y_state_raw[state_mask > 0.5]
    if len(feas_state) == 0:
        state_mean = np.zeros(len(state_names), dtype=np.float32)
        state_std = np.ones(len(state_names), dtype=np.float32)
        state_min = np.zeros(len(state_names), dtype=np.float32)
        state_max = np.ones(len(state_names), dtype=np.float32)
    else:
        state_mean = feas_state.mean(axis=0).astype(np.float32)
        state_std = _safe_std(feas_state, axis=0).astype(np.float32)
        smin = feas_state.min(axis=0)
        smax = feas_state.max(axis=0)
        pad = np.maximum((smax - smin) * 0.05, 1e-3)
        state_min = (smin - pad).astype(np.float32)
        state_max = (smax + pad).astype(np.float32)

    y_state_norm = np.zeros_like(y_state_raw)
    if len(feas_state) > 0:
        y_state_norm[state_mask > 0.5] = (y_state_raw[state_mask > 0.5] - state_mean) / state_std

    return MultiCaseDataset(
        case_id=case_id,
        input_names=input_names,
        state_names=state_names,
        X_raw=X_raw,
        X_norm=X_norm,
        y_cls=y_cls,
        y_state_raw=y_state_raw,
        y_state_norm=y_state_norm,
        state_mask=state_mask,
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        state_mean=state_mean,
        state_std=state_std,
        state_min=state_min,
        state_max=state_max,
        closure=closure,
    )


def build_wb2_dataset(seed: int = 42, n_samples: int = 8000) -> MultiCaseDataset:
    X_norm, y, meta = generate_WB2_data(n_samples=n_samples, seed=seed)
    X_raw = np.column_stack([meta["P_raw"], meta["Q_raw"]]).astype(np.float32)

    state_names = ["PG1", "QG1", "V2", "theta2", "line_loss"]
    y_state_raw = np.column_stack([
        np.asarray(meta["PG1"], dtype=np.float32),
        np.asarray(meta["QG1"], dtype=np.float32),
        np.asarray(meta["V2"], dtype=np.float32),
        np.asarray(meta["theta2"], dtype=np.float32),
        np.asarray(meta["line_loss"], dtype=np.float32),
    ])
    state_mask = (~np.isnan(y_state_raw).any(axis=1)).astype(np.float32)
    y_state_raw = np.nan_to_num(y_state_raw, nan=0.0)

    closure = ClosureConfig(
        target_name="PG1",
        loss_name="line_loss",
        constant_mw=0.0,
        input_terms=[(0, 1.0)],  # PG1 = P2_load + loss
        monotonic_terms=[(0, +1)],
    )
    x_mean = np.array(meta["X_mean"], dtype=np.float32)
    x_std = np.array(meta["X_std"], dtype=np.float32)

    return _finalize_dataset(
        case_id="WB2",
        input_names=["P2_load", "Q2_load"],
        state_names=state_names,
        X_raw=X_raw,
        y_cls=np.asarray(y, dtype=np.float32),
        y_state_raw=y_state_raw,
        state_mask=state_mask,
        x_mean=x_mean,
        x_std=x_std,
        closure=closure,
    )


def build_wb5_dataset(data_dir: Path, seed: int = 42, max_feasible: int = 12000) -> MultiCaseDataset:
    path = data_dir / "5节点数据.csv"
    df = pd.read_csv(path)
    pg1 = df["PG1"].to_numpy(dtype=np.float32)
    pg5 = df["PG5"].to_numpy(dtype=np.float32)
    loss = df["loss"].to_numpy(dtype=np.float32)

    rng = np.random.default_rng(seed)
    if len(df) > max_feasible:
        pick = rng.choice(len(df), size=max_feasible, replace=False)
        pg1, pg5, loss = pg1[pick], pg5[pick], loss[pick]

    X_feas = np.column_stack([pg1, pg5]).astype(np.float32)
    y_feas = np.ones(len(X_feas), dtype=np.float32)

    pg1_vals = np.arange(0.0, 700.0 + 0.5, 0.5, dtype=np.float32)
    pg5_vals = np.arange(0.0, 400.0 + 1.0, 1.0, dtype=np.float32)
    secure_mask = np.zeros((len(pg5_vals), len(pg1_vals)), dtype=bool)
    ix = np.rint((pg1 - pg1_vals[0]) / 0.5).astype(int)
    iy = np.rint((pg5 - pg5_vals[0]) / 1.0).astype(int)
    valid = (ix >= 0) & (ix < len(pg1_vals)) & (iy >= 0) & (iy < len(pg5_vals))
    secure_mask[iy[valid], ix[valid]] = True

    insecure_idx = np.argwhere(~secure_mask)
    n_bg = min(2 * len(X_feas), len(insecure_idx))
    pick_bg = insecure_idx[rng.choice(len(insecure_idx), size=n_bg, replace=False)]
    X_bg = np.column_stack([pg1_vals[pick_bg[:, 1]], pg5_vals[pick_bg[:, 0]]]).astype(np.float32)
    y_bg = np.zeros(len(X_bg), dtype=np.float32)

    X_raw = np.vstack([X_feas, X_bg]).astype(np.float32)
    y_cls = np.concatenate([y_feas, y_bg]).astype(np.float32)

    state_names = ["loss"]
    y_state_raw = np.zeros((len(X_raw), 1), dtype=np.float32)
    state_mask = np.zeros(len(X_raw), dtype=np.float32)
    y_state_raw[: len(X_feas), 0] = loss
    state_mask[: len(X_feas)] = 1.0

    closure = ClosureConfig(
        target_name=None,
        loss_name="loss",
        constant_mw=325.0,
        input_terms=[],
        monotonic_terms=[],
    )

    return _finalize_dataset(
        case_id="WB5",
        input_names=["PG1", "PG5"],
        state_names=state_names,
        X_raw=X_raw,
        y_cls=y_cls,
        y_state_raw=y_state_raw,
        state_mask=state_mask,
        x_mean=np.array([350.0, 200.0], dtype=np.float32),
        x_std=np.array([350.0, 200.0], dtype=np.float32),
        closure=closure,
    )


def build_case9mod_dataset(data_dir: Path, seed: int = 42) -> MultiCaseDataset:
    path = data_dir / "ac_opf_9results.csv"
    df = pd.read_csv(path)
    req = [
        "p2_mw", "p3_mw", "p1_mw", "q1_mvar", "q2_mvar", "q3_mvar",
        "v1_pu", "v2_pu", "v3_pu", "v4_pu", "v5_pu", "v6_pu", "v7_pu", "v8_pu", "v9_pu",
        "theta1_deg", "theta2_deg", "theta3_deg", "theta4_deg", "theta5_deg", "theta6_deg", "theta7_deg", "theta8_deg", "theta9_deg",
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in case9mod CSV: {missing}")

    p2_vals = np.linspace(10.0, 300.0, 300, dtype=np.float32)
    p3_vals = np.linspace(10.0, 270.0, 300, dtype=np.float32)
    d2 = float(p2_vals[1] - p2_vals[0])
    d3 = float(p3_vals[1] - p3_vals[0])

    ix = np.rint((df["p2_mw"].to_numpy(dtype=np.float64) - p2_vals[0]) / d2).astype(int)
    iy = np.rint((df["p3_mw"].to_numpy(dtype=np.float64) - p3_vals[0]) / d3).astype(int)
    valid = (ix >= 0) & (ix < len(p2_vals)) & (iy >= 0) & (iy < len(p3_vals))
    df = df.loc[valid].copy().reset_index(drop=True)
    ix = ix[valid]
    iy = iy[valid]
    df["_ix"] = ix
    df["_iy"] = iy
    df = df.drop_duplicates(subset=["_ix", "_iy"], keep="first").reset_index(drop=True)

    ix = df["_ix"].to_numpy(dtype=int)
    iy = df["_iy"].to_numpy(dtype=int)
    secure_mask = np.zeros((len(p3_vals), len(p2_vals)), dtype=bool)
    secure_mask[iy, ix] = True

    p_loss = (df["p1_mw"].to_numpy(dtype=np.float32)
              + df["p2_mw"].to_numpy(dtype=np.float32)
              + df["p3_mw"].to_numpy(dtype=np.float32)
              - df["total_load"].to_numpy(dtype=np.float32))

    state_names = [
        "p1_mw", "p_loss_mw", "q1_mvar", "q2_mvar", "q3_mvar",
        "v1_pu", "v2_pu", "v3_pu", "v4_pu", "v5_pu", "v6_pu", "v7_pu", "v8_pu", "v9_pu",
        "theta1_deg", "theta2_deg", "theta3_deg", "theta4_deg", "theta5_deg", "theta6_deg", "theta7_deg", "theta8_deg", "theta9_deg",
    ]

    X_feas = np.column_stack([p2_vals[ix], p3_vals[iy]]).astype(np.float32)
    y_feas = np.ones(len(X_feas), dtype=np.float32)
    S_feas = np.column_stack([
        df["p1_mw"].to_numpy(dtype=np.float32),
        p_loss.astype(np.float32),
        df["q1_mvar"].to_numpy(dtype=np.float32),
        df["q2_mvar"].to_numpy(dtype=np.float32),
        df["q3_mvar"].to_numpy(dtype=np.float32),
        df["v1_pu"].to_numpy(dtype=np.float32),
        df["v2_pu"].to_numpy(dtype=np.float32),
        df["v3_pu"].to_numpy(dtype=np.float32),
        df["v4_pu"].to_numpy(dtype=np.float32),
        df["v5_pu"].to_numpy(dtype=np.float32),
        df["v6_pu"].to_numpy(dtype=np.float32),
        df["v7_pu"].to_numpy(dtype=np.float32),
        df["v8_pu"].to_numpy(dtype=np.float32),
        df["v9_pu"].to_numpy(dtype=np.float32),
        df["theta1_deg"].to_numpy(dtype=np.float32),
        df["theta2_deg"].to_numpy(dtype=np.float32),
        df["theta3_deg"].to_numpy(dtype=np.float32),
        df["theta4_deg"].to_numpy(dtype=np.float32),
        df["theta5_deg"].to_numpy(dtype=np.float32),
        df["theta6_deg"].to_numpy(dtype=np.float32),
        df["theta7_deg"].to_numpy(dtype=np.float32),
        df["theta8_deg"].to_numpy(dtype=np.float32),
        df["theta9_deg"].to_numpy(dtype=np.float32),
    ]).astype(np.float32)

    rng = np.random.default_rng(seed)
    insecure_idx = np.argwhere(~secure_mask)
    n_bg_grid = min(2 * len(X_feas), len(insecure_idx))
    pick_bg = insecure_idx[rng.choice(len(insecure_idx), size=n_bg_grid, replace=False)]
    X_bg_grid = np.column_stack([p2_vals[pick_bg[:, 1]], p3_vals[pick_bg[:, 0]]]).astype(np.float32)

    n_guard = min(len(X_feas), 3000)
    n_x = n_guard // 2
    n_y = n_guard - n_x
    X_guard_x = np.column_stack([rng.uniform(0.0, 10.0, size=n_x), rng.uniform(0.0, 180.0, size=n_x)]).astype(np.float32)
    X_guard_y = np.column_stack([rng.uniform(0.0, 180.0, size=n_y), rng.uniform(0.0, 10.0, size=n_y)]).astype(np.float32)
    X_guard = np.vstack([X_guard_x, X_guard_y]).astype(np.float32)

    X_bg = np.vstack([X_bg_grid, X_guard]).astype(np.float32)
    y_bg = np.zeros(len(X_bg), dtype=np.float32)

    X_raw = np.vstack([X_feas, X_bg]).astype(np.float32)
    y_cls = np.concatenate([y_feas, y_bg]).astype(np.float32)

    y_state_raw = np.zeros((len(X_raw), len(state_names)), dtype=np.float32)
    state_mask = np.zeros(len(X_raw), dtype=np.float32)
    y_state_raw[: len(X_feas)] = S_feas
    state_mask[: len(X_feas)] = 1.0

    total_load = float(df["total_load"].iloc[0]) if "total_load" in df.columns else 189.0
    closure = ClosureConfig(
        target_name="p1_mw",
        loss_name="p_loss_mw",
        constant_mw=total_load,
        input_terms=[(0, -1.0), (1, -1.0)],
        monotonic_terms=[(0, -1), (1, -1)],
    )

    return _finalize_dataset(
        case_id="case9mod",
        input_names=["p2_mw", "p3_mw"],
        state_names=state_names,
        X_raw=X_raw,
        y_cls=y_cls,
        y_state_raw=y_state_raw,
        state_mask=state_mask,
        x_mean=np.array([163.0, 85.0], dtype=np.float32),
        x_std=np.array([145.0, 130.0], dtype=np.float32),
        closure=closure,
    )


def build_lmbm3_dataset(
    data_dir: Path,
    label: str,
    seed: int = 42,
    max_feasible: int = 8000,
) -> MultiCaseDataset:
    if label == "LMBM3_lf1p490":
        path = data_dir / "lmbm3 负荷1.490.csv"
    elif label == "LMBM3_lf1p500":
        path = data_dir / "lmbm3_feasible_points_v2_optimized.csv"
    else:
        raise ValueError(f"Unknown LMBM3 label: {label}")

    df = pd.read_csv(path)
    rng = np.random.default_rng(seed)
    if len(df) > max_feasible:
        df = df.sample(n=max_feasible, random_state=seed).reset_index(drop=True)

    # Unified column names
    v_cols = ["V1", "V2", "V3"] if all(c in df.columns for c in ["V1", "V2", "V3"]) else ["Vm1", "Vm2", "Vm3"]
    a_cols = ["Va1", "Va2", "Va3"] if all(c in df.columns for c in ["Va1", "Va2", "Va3"]) else ["Va1_deg", "Va2_deg", "Va3_deg"]
    loss_col = "loss" if "loss" in df.columns else "Ploss_MW"

    pg1 = df["PG1"].to_numpy(dtype=np.float32)
    pg2 = df["PG2"].to_numpy(dtype=np.float32)
    pg3 = df["PG3"].to_numpy(dtype=np.float32)
    q1 = df["QG1"].to_numpy(dtype=np.float32)
    q2 = df["QG2"].to_numpy(dtype=np.float32)
    q3 = df["QG3"].to_numpy(dtype=np.float32)
    v1 = df[v_cols[0]].to_numpy(dtype=np.float32)
    v2 = df[v_cols[1]].to_numpy(dtype=np.float32)
    v3 = df[v_cols[2]].to_numpy(dtype=np.float32)
    a1 = df[a_cols[0]].to_numpy(dtype=np.float32)
    a2 = df[a_cols[1]].to_numpy(dtype=np.float32)
    a3 = df[a_cols[2]].to_numpy(dtype=np.float32)
    loss = df[loss_col].to_numpy(dtype=np.float32)

    X_feas = np.column_stack([pg1, pg2]).astype(np.float32)
    y_feas = np.ones(len(X_feas), dtype=np.float32)

    S_feas = np.column_stack([pg3, q1, q2, q3, v1, v2, v3, a1, a2, a3, loss]).astype(np.float32)
    state_names = ["PG3", "QG1", "QG2", "QG3", "V1", "V2", "V3", "Va1", "Va2", "Va3", "loss"]

    n_bg = 2 * len(X_feas)
    x1_bg = rng.uniform(pg1.min() * 0.8, pg1.max() * 1.2, n_bg).astype(np.float32)
    x2_bg = rng.uniform(pg2.min() * 0.8, pg2.max() * 1.2, n_bg).astype(np.float32)
    X_bg_raw = np.column_stack([x1_bg, x2_bg]).astype(np.float32)

    tree = KDTree(X_feas)
    dx = max((pg1.max() - pg1.min()) / 30.0, 1e-6)
    dy = max((pg2.max() - pg2.min()) / 30.0, 1e-6)
    radius = 2.5 * np.sqrt(dx * dx + dy * dy)
    counts = tree.query_radius(X_bg_raw, r=radius, count_only=True)
    X_bg = X_bg_raw[counts == 0][:n_bg]
    y_bg = np.zeros(len(X_bg), dtype=np.float32)

    X_raw = np.vstack([X_feas, X_bg]).astype(np.float32)
    y_cls = np.concatenate([y_feas, y_bg]).astype(np.float32)

    y_state_raw = np.zeros((len(X_raw), len(state_names)), dtype=np.float32)
    state_mask = np.zeros(len(X_raw), dtype=np.float32)
    y_state_raw[: len(X_feas)] = S_feas
    state_mask[: len(X_feas)] = 1.0

    total_load = float(np.mean(pg1 + pg2 + pg3 - loss))
    closure = ClosureConfig(
        target_name="PG3",
        loss_name="loss",
        constant_mw=total_load,
        input_terms=[(0, -1.0), (1, -1.0)],
        monotonic_terms=[(0, -1), (1, -1)],
    )

    return _finalize_dataset(
        case_id=label,
        input_names=["PG1", "PG2"],
        state_names=state_names,
        X_raw=X_raw,
        y_cls=y_cls,
        y_state_raw=y_state_raw,
        state_mask=state_mask,
        x_mean=X_raw.mean(axis=0).astype(np.float32),
        x_std=_safe_std(X_raw, axis=0).astype(np.float32),
        closure=closure,
    )


def build_dataset(case_id: str, data_dir: Path, seed: int = 42) -> MultiCaseDataset:
    if case_id == "WB2":
        return build_wb2_dataset(seed=seed)
    if case_id == "WB5":
        return build_wb5_dataset(data_dir=data_dir, seed=seed)
    if case_id == "case9mod":
        return build_case9mod_dataset(data_dir=data_dir, seed=seed)
    if case_id in ("LMBM3_lf1p490", "LMBM3_lf1p500"):
        return build_lmbm3_dataset(data_dir=data_dir, label=case_id, seed=seed)
    raise ValueError(f"Unsupported case_id: {case_id}")
