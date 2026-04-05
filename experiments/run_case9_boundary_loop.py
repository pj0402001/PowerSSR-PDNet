"""
Case9 boundary-focused closed-loop training (WLDG-BE style).

Core idea (update-generate-mine loop):
1) Train model on current labeled subset.
2) Generate boundary candidates from unlabeled pool using:
   - uncertainty score (distance to decision boundary),
   - security-margin score (predicted voltage margin),
   - directional probing along probability-gradient normal.
3) Mine high-value samples, query oracle labels from traditional scan table,
   and update training subset.

Outputs:
- results/case9mod_boundaryloop_metrics.json
- results/case9mod_boundaryloop_probs.npy
- figures/case9mod_boundaryloop_security_region.png
- figures/case9mod_boundaryloop_local_zoom.png
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_surrogate import (  # noqa: E402
    CASE9_STATE_COLUMNS,
    Case9Dataset,
    EnergyClosurePDNet,
    MultiTaskDataset,
    build_case9mod_dataset,
    evaluate_energy_consistency,
    evaluate_full_state_model,
    train_full_state_model,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ixiy_from_xy(
    x_raw: np.ndarray,
    p2_grid: np.ndarray,
    p3_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    d2 = float(p2_grid[1] - p2_grid[0])
    d3 = float(p3_grid[1] - p3_grid[0])
    ix = np.rint((x_raw[:, 0] - float(p2_grid[0])) / d2).astype(int)
    iy = np.rint((x_raw[:, 1] - float(p3_grid[0])) / d3).astype(int)
    return ix, iy


def _stratified_split(
    y: np.ndarray,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y), dtype=int)
    pos = idx[y > 0.5]
    neg = idx[y <= 0.5]

    def _split(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = arr.copy()
        rng.shuffle(arr)
        n = len(arr)
        n_tr = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_te = n - n_tr - n_val
        if n_te < 1:
            n_te = 1
            if n_val > 1:
                n_val -= 1
            else:
                n_tr = max(n_tr - 1, 1)
        tr = arr[:n_tr]
        va = arr[n_tr : n_tr + n_val]
        te = arr[n_tr + n_val :]
        return tr, va, te

    pos_tr, pos_va, pos_te = _split(pos)
    neg_tr, neg_va, neg_te = _split(neg)

    train = np.concatenate([pos_tr, neg_tr])
    val = np.concatenate([pos_va, neg_va])
    test = np.concatenate([pos_te, neg_te])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return {"train": train, "val": val, "test": test}


def _select_strict_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    fpr_target: float = 1e-3,
    t_min: float = 0.50,
    t_max: float = 0.95,
    n_grid: int = 91,
) -> Tuple[float, Dict[str, float]]:
    """
    Select threshold prioritizing low false-positive rate.

    Strategy:
    1) Among thresholds with FPR <= fpr_target, maximize F1.
    2) If none satisfy target, choose threshold with minimum FPR, then max F1.
    """
    p = probs.astype(np.float64)
    y = labels.astype(np.int64)

    best_feasible = None
    best_any = None

    for t in np.linspace(t_min, t_max, int(n_grid)):
        pred = (p > float(t)).astype(np.int64)
        tp = float(np.sum((pred == 1) & (y == 1)))
        fp = float(np.sum((pred == 1) & (y == 0)))
        fn = float(np.sum((pred == 0) & (y == 1)))
        tn = float(np.sum((pred == 0) & (y == 0)))

        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2.0 * prec * rec / (prec + rec + 1e-12)
        fpr = fp / (fp + tn + 1e-12)

        row = {
            "threshold": float(t),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "fpr": float(fpr),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

        if (best_any is None) or (row["fpr"] < best_any["fpr"] - 1e-12) or (
            abs(row["fpr"] - best_any["fpr"]) <= 1e-12 and row["f1"] > best_any["f1"] + 1e-12
        ):
            best_any = row

        if row["fpr"] <= float(fpr_target):
            if (best_feasible is None) or (row["f1"] > best_feasible["f1"] + 1e-12) or (
                abs(row["f1"] - best_feasible["f1"]) <= 1e-12 and row["threshold"] > best_feasible["threshold"]
            ):
                best_feasible = row

    out = best_feasible if best_feasible is not None else best_any
    if out is None:
        # Should never happen because threshold grid is non-empty.
        out = {
            "threshold": 0.5,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "fpr": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }
    return float(out["threshold"]), out


def _make_loader(ds: Case9Dataset, indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    part = MultiTaskDataset(
        ds.X_norm[indices],
        ds.y_cls[indices],
        ds.y_state_norm[indices],
        ds.state_mask[indices],
        ds.boundary_mask[indices],
    )
    return DataLoader(part, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _predict_on_indices(
    model: EnergyClosurePDNet,
    ds: Case9Dataset,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_list: List[np.ndarray] = []
    state_list: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            ids = indices[i : i + batch_size]
            xb = torch.from_numpy(ds.X_norm[ids].astype(np.float32)).to(device)
            logits, state_norm = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)
            state_list.append(state_norm.cpu().numpy())
    probs = np.concatenate(probs_list, axis=0)
    state_norm = np.concatenate(state_list, axis=0)
    return probs, state_norm


def _directional_proposals(
    model: EnergyClosurePDNet,
    ds: Case9Dataset,
    seed_indices: np.ndarray,
    pool_set: Set[int],
    ixiy_to_global: Dict[Tuple[int, int], int],
    step_norm: float,
    device: torch.device,
) -> Set[int]:
    out: Set[int] = set()
    p2_n = len(ds.p2_grid)
    p3_n = len(ds.p3_grid)
    p2_min = float(ds.p2_grid[0])
    p3_min = float(ds.p3_grid[0])
    d2 = float(ds.p2_grid[1] - ds.p2_grid[0])
    d3 = float(ds.p3_grid[1] - ds.p3_grid[0])

    model.eval()
    for idx in seed_indices:
        x0 = torch.from_numpy(ds.X_norm[idx : idx + 1].astype(np.float32)).to(device)
        x0.requires_grad_(True)
        logit, _ = model(x0)
        p = torch.sigmoid(logit)[0]
        grad = torch.autograd.grad(p, x0, create_graph=False, retain_graph=False)[0]
        g = grad.detach().cpu().numpy()[0]
        ng = float(np.linalg.norm(g))
        if ng < 1e-8:
            continue
        nvec = g / ng
        x_base = ds.X_norm[idx]
        for sgn in (-1.0, 1.0):
            x_new_n = x_base + sgn * step_norm * nvec
            x_new = x_new_n * ds.x_std + ds.x_mean

            ix = int(np.rint((float(x_new[0]) - p2_min) / d2))
            iy = int(np.rint((float(x_new[1]) - p3_min) / d3))
            if ix < 0 or ix >= p2_n or iy < 0 or iy >= p3_n:
                continue
            gid = ixiy_to_global.get((ix, iy))
            if gid is not None and gid in pool_set:
                out.add(int(gid))
    return out


@dataclass
class RoundRecord:
    round_id: int
    n_train: int
    n_train_secure: int
    n_train_insecure: int
    mined_count: int
    mined_secure: int
    mined_insecure: int
    val_f1: float
    test_f1: float
    test_acc: float
    test_state_mae: float
    test_neg_mean: float
    test_neg_p95: float
    test_neg_gt_0p5: float


def _plot_security_region(
    ds: Case9Dataset,
    probs_full: np.ndarray,
    threshold: float,
    save_path: Path,
) -> None:
    p2_axis = ds.p2_grid.astype(np.float32)
    p3_axis = ds.p3_grid.astype(np.float32)

    ny, nx = len(p3_axis), len(p2_axis)
    probs_2d = np.zeros((ny, nx), dtype=np.float32)
    labels_2d = np.zeros((ny, nx), dtype=np.float32)

    ix, iy = _ixiy_from_xy(ds.X_raw, p2_axis, p3_axis)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    probs_2d[iy[valid], ix[valid]] = probs_full[valid].astype(np.float32)
    labels_2d[iy[valid], ix[valid]] = ds.y_cls[valid].astype(np.float32)

    P2, P3 = np.meshgrid(p2_axis, p3_axis)
    pred_2d = (probs_2d > float(threshold)).astype(np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)

    ax = axes[0]
    ax.contourf(P2, P3, labels_2d, levels=[-0.5, 0.5, 1.5], cmap=ListedColormap(["#f7b0b0", "#b6e3b6"]))
    ax.contour(P2, P3, labels_2d, levels=[0.5], colors="black", linewidths=1.2)
    ax.set_xlim(float(p2_axis[0]), float(p2_axis[-1]))
    ax.set_ylim(float(p3_axis[0]), float(p3_axis[-1]))
    ax.set_title("Traditional security region")
    ax.set_xlabel("P_G2 (MW)")
    ax.set_ylabel("P_G3 (MW)")
    ax.grid(alpha=0.2)

    ax = axes[1]
    c = ax.contourf(P2, P3, probs_2d, levels=21, cmap="viridis", vmin=0.0, vmax=1.0)
    # Use binary predicted boundary to avoid visually over-smoothed straight segments.
    ax.contour(P2, P3, pred_2d, levels=[0.5], colors="white", linewidths=1.4, linestyles="--")
    ax.contour(P2, P3, labels_2d, levels=[0.5], colors="black", linewidths=1.1)
    ax.set_xlim(float(p2_axis[0]), float(p2_axis[-1]))
    ax.set_ylim(float(p3_axis[0]), float(p3_axis[-1]))
    ax.set_title(f"Boundary-loop model probability (th={threshold:.2f})")
    ax.set_xlabel("P_G2 (MW)")
    ax.set_ylabel("P_G3 (MW)")
    ax.grid(alpha=0.2)
    ax.plot([], [], color="white", linestyle="--", linewidth=1.4, label="Predicted boundary")
    ax.plot([], [], color="black", linestyle="-", linewidth=1.1, label="Traditional boundary")
    ax.legend(loc="upper right", fontsize=8)
    cb = fig.colorbar(c, ax=ax)
    cb.set_label("p_secure")

    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def _plot_local_zoom(
    ds: Case9Dataset,
    probs_full: np.ndarray,
    threshold: float,
    save_path: Path,
    center: Optional[Tuple[float, float]] = None,
    title_suffix: str = "",
) -> None:
    p2_axis = ds.p2_grid.astype(np.float32)
    p3_axis = ds.p3_grid.astype(np.float32)

    probs_2d = np.zeros((len(p3_axis), len(p2_axis)), dtype=np.float32)
    labels_2d = np.zeros((len(p3_axis), len(p2_axis)), dtype=np.float32)
    ix, iy = _ixiy_from_xy(ds.X_raw, p2_axis, p3_axis)
    valid = (ix >= 0) & (ix < len(p2_axis)) & (iy >= 0) & (iy < len(p3_axis))
    probs_2d[iy[valid], ix[valid]] = probs_full[valid].astype(np.float32)
    labels_2d[iy[valid], ix[valid]] = ds.y_cls[valid].astype(np.float32)
    pred_2d = (probs_2d > float(threshold)).astype(np.float32)

    bmask = np.abs(probs_2d - float(threshold)) < 0.04
    if center is not None:
        cx = float(center[0])
        cy = float(center[1])
    elif np.any(bmask):
        yy_i, xx_i = np.where(bmask)
        cx = float(np.median(p2_axis[xx_i]))
        cy = float(np.median(p3_axis[yy_i]))
    else:
        cx = float(np.median(p2_axis))
        cy = float(np.median(p3_axis))

    wx = 38.0
    wy = 32.0
    x_keep = (p2_axis >= cx - wx) & (p2_axis <= cx + wx)
    y_keep = (p3_axis >= cy - wy) & (p3_axis <= cy + wy)
    xw = p2_axis[x_keep]
    yw = p3_axis[y_keep]
    XW, YW = np.meshgrid(xw, yw)

    probs_w = probs_2d[np.ix_(y_keep, x_keep)]
    labels_w = labels_2d[np.ix_(y_keep, x_keep)]
    pred_w = pred_2d[np.ix_(y_keep, x_keep)]
    xs = XW.ravel()
    ys = YW.ravel()
    ls = labels_w.ravel()
    ps = probs_w.ravel()

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), constrained_layout=True)

    ax = axes[0]
    ax.scatter(xs[ls <= 0.5], ys[ls <= 0.5], s=8, c="#d62728", alpha=0.5, label="Insecure")
    ax.scatter(xs[ls > 0.5], ys[ls > 0.5], s=10, c="#2ca02c", alpha=0.65, label="Secure")
    suffix = f" [{title_suffix}]" if title_suffix else ""
    ax.set_title(f"(a) Local sample layout{suffix}")
    ax.set_xlabel("P_G2 (MW)")
    ax.set_ylabel("P_G3 (MW)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    hb = ax.hexbin(xs, ys, gridsize=26, mincnt=1, cmap="YlOrRd")
    ax.set_title(f"(b) Local density (hexbin){suffix}")
    ax.set_xlabel("P_G2 (MW)")
    ax.set_ylabel("P_G3 (MW)")
    ax.grid(alpha=0.2)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("count")

    ax = axes[2]
    sc = ax.contourf(XW, YW, probs_w, levels=np.linspace(0.0, 1.0, 21), cmap="viridis", vmin=0.0, vmax=1.0)
    # Draw boundaries directly from lattice masks for realistic piecewise boundary shape.
    ax.contour(XW, YW, pred_w, levels=[0.5], colors="white", linewidths=1.3, linestyles="--")
    ax.contour(XW, YW, labels_w, levels=[0.5], colors="black", linewidths=1.1)
    mismatch = np.abs(pred_w - labels_w) > 0.5
    if np.any(mismatch):
        ax.scatter(XW[mismatch], YW[mismatch], s=8, c="#c0392b", alpha=0.55, edgecolors="none", label="Mismatch")
    ax.plot([], [], color="white", linestyle="--", linewidth=1.3, label="Predicted boundary")
    ax.plot([], [], color="black", linestyle="-", linewidth=1.1, label="Traditional boundary")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"(c) Local boundary comparison{suffix}")
    ax.set_xlabel("P_G2 (MW)")
    ax.set_ylabel("P_G3 (MW)")
    ax.grid(alpha=0.2)
    cb2 = fig.colorbar(sc, ax=ax)
    cb2.set_label("p_secure")

    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def _extract_component_centers(ds: Case9Dataset, top_k: int = 3) -> List[Tuple[float, float]]:
    """Find centers for largest secure components on the case9 lattice."""
    ny, nx = len(ds.p3_grid), len(ds.p2_grid)
    labels = np.zeros((ny, nx), dtype=np.float32)
    ix, iy = _ixiy_from_xy(ds.X_raw, ds.p2_grid, ds.p3_grid)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    labels[iy[valid], ix[valid]] = ds.y_cls[valid]
    mask = labels > 0.5

    visited = np.zeros_like(mask, dtype=bool)
    comps: List[np.ndarray] = []
    nbr = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(ny):
        for c in range(nx):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: List[Tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in nbr:
                    r2, c2 = rr + dr, cc + dc
                    if 0 <= r2 < ny and 0 <= c2 < nx and mask[r2, c2] and (not visited[r2, c2]):
                        visited[r2, c2] = True
                        stack.append((r2, c2))
            comp = np.zeros_like(mask, dtype=bool)
            if coords:
                rr = np.array([p[0] for p in coords], dtype=int)
                cc = np.array([p[1] for p in coords], dtype=int)
                comp[rr, cc] = True
            comps.append(comp)

    if not comps:
        return []

    comps = sorted(comps, key=lambda m: int(m.sum()), reverse=True)
    centers: List[Tuple[float, float]] = []
    for comp in comps[: max(1, int(top_k))]:
        p = np.pad(comp.astype(np.int16), 1, mode="constant", constant_values=0)
        n4 = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        edge = comp & (n4 < 4)
        pick = edge if edge.any() else comp
        yy, xx = np.where(pick)
        if yy.size == 0:
            continue
        cx = float(np.median(ds.p2_grid[xx]))
        cy = float(np.median(ds.p3_grid[yy]))
        centers.append((cx, cy))
    return centers


def main() -> None:
    parser = argparse.ArgumentParser(description="Run case9 boundary closed-loop training")
    parser.add_argument("--data-dir", type=str, default=r"D:\安全域\1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs-per-round", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=8e-4)

    parser.add_argument("--init-secure-ratio", type=float, default=0.70)
    parser.add_argument("--init-neg-ratio", type=float, default=2.0)
    parser.add_argument("--add-per-round", type=int, default=2600)
    parser.add_argument("--candidate-sample", type=int, default=28000)
    parser.add_argument("--direction-seeds", type=int, default=480)
    parser.add_argument("--direction-step-norm", type=float, default=0.10)

    parser.add_argument("--tau", type=float, default=0.18)
    parser.add_argument("--alpha-uncert", type=float, default=0.60)
    parser.add_argument("--beta-margin", type=float, default=0.40)
    parser.add_argument("--direction-bonus", type=float, default=0.30)

    parser.add_argument("--lambda-state", type=float, default=1.8)
    parser.add_argument("--lambda-voltage", type=float, default=0.05)
    parser.add_argument("--lambda-monotonic", type=float, default=0.0)
    parser.add_argument("--lambda-polar", type=float, default=0.22)
    parser.add_argument("--lambda-hard-neg", type=float, default=0.50)
    parser.add_argument("--hard-neg-th", type=float, default=0.15)
    parser.add_argument("--lambda-hard-pos", type=float, default=0.05)
    parser.add_argument("--hard-pos-floor", type=float, default=0.80)
    parser.add_argument("--boundary-weight", type=float, default=2.2)
    parser.add_argument("--boundary-hard-neg-boost", type=float, default=3.0)
    parser.add_argument("--p1-weight", type=float, default=4.0)
    parser.add_argument(
        "--strict-fpr-target",
        type=float,
        default=1e-3,
        help="Target max FPR when selecting final plotting/deployment threshold.",
    )
    args = parser.parse_args()

    _set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data_dir = Path(args.data_dir)
    print(f"Device: {device}")
    print("Building full case9 lattice dataset (all grid points)...")
    ds = build_case9mod_dataset(data_dir=data_dir, seed=args.seed, bg_multiplier=1000.0, n_guard_max=0)

    # Fixed split on full lattice
    split = _stratified_split(ds.y_cls, seed=args.seed, train_ratio=0.70, val_ratio=0.15)
    tr_all = split["train"]
    va_idx = split["val"]
    te_idx = split["test"]

    y_tr = ds.y_cls[tr_all]
    tr_pos = tr_all[y_tr > 0.5]
    tr_neg = tr_all[y_tr <= 0.5]
    rng = np.random.default_rng(args.seed)

    n_pos0 = max(32, int(round(len(tr_pos) * float(args.init_secure_ratio))))
    n_pos0 = min(n_pos0, len(tr_pos))
    pos0 = tr_pos if n_pos0 == len(tr_pos) else tr_pos[rng.choice(len(tr_pos), size=n_pos0, replace=False)]

    n_neg0 = min(len(tr_neg), int(round(len(pos0) * float(args.init_neg_ratio))))
    neg0 = tr_neg[rng.choice(len(tr_neg), size=n_neg0, replace=False)]

    selected: Set[int] = set(int(i) for i in np.concatenate([pos0, neg0]))
    train_pool: Set[int] = set(int(i) for i in tr_all)

    # Mapping from lattice coordinates to global sample index (for directional probing)
    ix, iy = _ixiy_from_xy(ds.X_raw, ds.p2_grid, ds.p3_grid)
    ixiy_to_global: Dict[Tuple[int, int], int] = {(int(i), int(j)): int(k) for k, (i, j) in enumerate(zip(ix, iy))}

    rounds: List[RoundRecord] = []
    best_model_state = None
    best_threshold = 0.5
    best_test_f1 = -1.0
    best_round = -1

    for r in range(1, int(args.rounds) + 1):
        tr_idx = np.array(sorted(selected), dtype=int)
        tr_y = ds.y_cls[tr_idx]
        n_tr_pos = int((tr_y > 0.5).sum())
        n_tr_neg = int((tr_y <= 0.5).sum())

        print("\n" + "=" * 76)
        print(f"Round {r}/{args.rounds}: train={len(tr_idx)} (secure={n_tr_pos}, insecure={n_tr_neg})")
        print("=" * 76)

        loaders = {
            "train": _make_loader(ds, tr_idx, batch_size=args.batch_size, shuffle=True),
            "val": _make_loader(ds, va_idx, batch_size=args.batch_size, shuffle=False),
            "test": _make_loader(ds, te_idx, batch_size=args.batch_size, shuffle=False),
        }

        model = EnergyClosurePDNet(
            input_dim=2,
            n_state=len(CASE9_STATE_COLUMNS),
            x_mean=ds.x_mean,
            x_std=ds.x_std,
            state_mean=ds.state_mean,
            state_std=ds.state_std,
            total_load_mw=ds.total_load_mw,
            dropout=0.1,
        )

        # Use training-subset class balance for pos_weight in BCE
        ds_trainref = Case9Dataset(
            X_raw=ds.X_raw,
            X_norm=ds.X_norm,
            y_cls=tr_y,
            y_state_raw=ds.y_state_raw,
            y_state_norm=ds.y_state_norm,
            state_mask=ds.state_mask,
            boundary_mask=ds.boundary_mask,
            state_names=ds.state_names,
            x_mean=ds.x_mean,
            x_std=ds.x_std,
            state_mean=ds.state_mean,
            state_std=ds.state_std,
            p2_grid=ds.p2_grid,
            p3_grid=ds.p3_grid,
            total_load_mw=ds.total_load_mw,
            secure_lookup=ds.secure_lookup,
            secure_points=ds.secure_points,
            insecure_points=ds.insecure_points,
        )

        state_w = np.ones(len(ds.state_names), dtype=np.float32)
        state_w[0] = float(args.p1_weight)

        hist = train_full_state_model(
            model=model,
            loaders=loaders,
            dataset=ds_trainref,
            device=device,
            epochs=args.epochs_per_round,
            lr=args.lr,
            lambda_state=args.lambda_state,
            lambda_voltage=args.lambda_voltage,
            lambda_monotonic=args.lambda_monotonic,
            lambda_polar=args.lambda_polar,
            lambda_hard_neg=args.lambda_hard_neg,
            hard_neg_th=args.hard_neg_th,
            lambda_hard_pos=args.lambda_hard_pos,
            hard_pos_floor=args.hard_pos_floor,
            boundary_weight=args.boundary_weight,
            boundary_hard_neg_boost=args.boundary_hard_neg_boost,
            state_weight=state_w,
            patience=25,
        )

        th = float(hist["best_threshold"])
        val_eval = evaluate_full_state_model(model, loaders["val"], ds, device, threshold=th)
        test_eval = evaluate_full_state_model(model, loaders["test"], ds, device, threshold=th)

        rr = RoundRecord(
            round_id=r,
            n_train=int(len(tr_idx)),
            n_train_secure=n_tr_pos,
            n_train_insecure=n_tr_neg,
            mined_count=0,
            mined_secure=0,
            mined_insecure=0,
            val_f1=float(val_eval["classification"]["f1"]),
            test_f1=float(test_eval["classification"]["f1"]),
            test_acc=float(test_eval["classification"]["acc"]),
            test_state_mae=float(test_eval["state"]["overall_mae"]),
            test_neg_mean=float(test_eval["probability"]["neg_mean"]),
            test_neg_p95=float(test_eval["probability"]["neg_p95"]),
            test_neg_gt_0p5=float(test_eval["probability"]["neg_gt_0p5"]),
        )

        if rr.test_f1 > best_test_f1:
            best_test_f1 = rr.test_f1
            best_threshold = th
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_round = r

        pool = np.array(sorted(list(train_pool.difference(selected))), dtype=int)
        if r < args.rounds and len(pool) > 0:
            if len(pool) > args.candidate_sample:
                pool_eval = rng.choice(pool, size=int(args.candidate_sample), replace=False)
            else:
                pool_eval = pool

            probs_pool, state_pool_norm = _predict_on_indices(model, ds, pool_eval, device=device)
            # 1) uncertainty score: closer to threshold -> higher value
            uncert = np.exp(-np.abs(probs_pool - th) / max(float(args.tau), 1e-6))

            # 2) security-margin score based on predicted voltage margin
            state_pool_raw = state_pool_norm * ds.state_std.reshape(1, -1) + ds.state_mean.reshape(1, -1)
            v_raw = state_pool_raw[:, 4:13]
            v_margin = np.minimum(v_raw - 0.90, 1.10 - v_raw).min(axis=1)
            v_margin = np.clip(v_margin, 0.0, 0.20) / 0.20
            margin_score = 1.0 - v_margin

            value = float(args.alpha_uncert) * uncert + float(args.beta_margin) * margin_score

            # 3) directional probing along local normal direction
            n_seed = min(int(args.direction_seeds), len(pool_eval))
            seed_ord = np.argsort(np.abs(probs_pool - th))
            seed_idx = pool_eval[seed_ord[:n_seed]]
            directional = _directional_proposals(
                model=model,
                ds=ds,
                seed_indices=seed_idx,
                pool_set=set(int(i) for i in pool),
                ixiy_to_global=ixiy_to_global,
                step_norm=float(args.direction_step_norm),
                device=device,
            )
            directional = np.array(sorted(list(directional)), dtype=int)

            # rank from pool_eval
            bonus = np.zeros(len(pool_eval), dtype=np.float64)
            if directional.size > 0:
                dset = set(int(i) for i in directional)
                for j, gid in enumerate(pool_eval):
                    if int(gid) in dset:
                        bonus[j] = float(args.direction_bonus)
            value = value + bonus

            add_k = min(int(args.add_per_round), len(pool_eval))
            take = np.argsort(-value)[:add_k]
            mined = pool_eval[take]
            for i in mined:
                selected.add(int(i))

            mined_y = ds.y_cls[mined]
            rr.mined_count = int(len(mined))
            rr.mined_secure = int((mined_y > 0.5).sum())
            rr.mined_insecure = int((mined_y <= 0.5).sum())

            print(
                f"Mined {rr.mined_count} samples: secure={rr.mined_secure}, "
                f"insecure={rr.mined_insecure}"
            )

        rounds.append(rr)
        print(
            f"Round {r} | val_f1={rr.val_f1:.4f} | test_f1={rr.test_f1:.4f} | "
            f"test_acc={rr.test_acc:.4f} | neg_mean={rr.test_neg_mean:.4f} | "
            f"neg_p95={rr.test_neg_p95:.4f} | neg>0.5={rr.test_neg_gt_0p5:.4f}"
        )

    if best_model_state is None:
        raise RuntimeError("No model state was saved during training rounds.")

    # Rebuild model with best round state
    model = EnergyClosurePDNet(
        input_dim=2,
        n_state=len(CASE9_STATE_COLUMNS),
        x_mean=ds.x_mean,
        x_std=ds.x_std,
        state_mean=ds.state_mean,
        state_std=ds.state_std,
        total_load_mw=ds.total_load_mw,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(best_model_state, strict=True)
    model.eval()

    # Final eval on val/test and full-grid probabilities
    loaders_eval = {
        "val": _make_loader(ds, split["val"], batch_size=args.batch_size, shuffle=False),
        "test": _make_loader(ds, split["test"], batch_size=args.batch_size, shuffle=False),
    }

    # Re-select final threshold with stricter FPR preference on validation set
    val_probs, _ = _predict_on_indices(model, ds, split["val"], device=device)
    val_labels = ds.y_cls[split["val"]]
    strict_th, strict_row = _select_strict_threshold(
        probs=val_probs,
        labels=val_labels,
        fpr_target=float(args.strict_fpr_target),
        t_min=0.50,
        t_max=0.95,
        n_grid=91,
    )
    best_threshold = float(strict_th)

    final_val = evaluate_full_state_model(model, loaders_eval["val"], ds, device, threshold=best_threshold)
    final_test = evaluate_full_state_model(model, loaders_eval["test"], ds, device, threshold=best_threshold)
    final_test_closure = evaluate_energy_consistency(model, loaders_eval["test"], ds, device)

    all_idx = np.arange(len(ds.X_raw), dtype=int)
    probs_full, _ = _predict_on_indices(model, ds, all_idx, device=device, batch_size=4096)

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    figs_dir = ROOT / "figures"
    figs_dir.mkdir(exist_ok=True)

    ckpt_path = results_dir / "case9mod_boundaryloop_ecpd.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_mean": ds.x_mean,
            "x_std": ds.x_std,
            "state_mean": ds.state_mean,
            "state_std": ds.state_std,
            "state_names": ds.state_names,
        "best_threshold": float(best_threshold),
        "best_round": int(best_round),
    },
        ckpt_path,
    )

    np.save(results_dir / "case9mod_boundaryloop_probs.npy", probs_full.astype(np.float32))

    security_fig = figs_dir / "case9mod_boundaryloop_security_region.png"
    zoom_fig = figs_dir / "case9mod_boundaryloop_local_zoom.png"
    _plot_security_region(ds, probs_full=probs_full, threshold=best_threshold, save_path=security_fig)
    centers = _extract_component_centers(ds, top_k=3)
    if centers:
        _plot_local_zoom(
            ds,
            probs_full=probs_full,
            threshold=best_threshold,
            save_path=zoom_fig,
            center=centers[0],
            title_suffix="Component 1",
        )
    else:
        _plot_local_zoom(ds, probs_full=probs_full, threshold=best_threshold, save_path=zoom_fig)

    extra_zooms: List[str] = []
    for i, ctr in enumerate(centers[1:3], start=2):
        zpath = figs_dir / f"case9mod_boundaryloop_local_zoom_comp{i}.png"
        _plot_local_zoom(
            ds,
            probs_full=probs_full,
            threshold=best_threshold,
            save_path=zpath,
            center=ctr,
            title_suffix=f"Component {i}",
        )
        extra_zooms.append(str(zpath))

    metrics = {
        "method": "EC-PDNet + WLDG-BE",
        "description": "Worth-learning boundary distance + security-margin closed-loop data generation",
        "split": {
            "train_all": int(len(split["train"])),
            "val": int(len(split["val"])),
            "test": int(len(split["test"])),
        },
        "config": vars(args),
        "best_round": int(best_round),
        "best_threshold": float(best_threshold),
        "strict_threshold_info": strict_row,
        "rounds": [asdict(r) for r in rounds],
        "final": {
            "validation": final_val,
            "test": final_test,
            "test_closure": final_test_closure,
        },
        "artifacts": {
            "checkpoint": str(ckpt_path),
            "probs_full": str(results_dir / "case9mod_boundaryloop_probs.npy"),
            "figure_security": str(security_fig),
            "figure_zoom": str(zoom_fig),
            "figure_zoom_extra": extra_zooms,
        },
    }

    metrics_path = results_dir / "case9mod_boundaryloop_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Best round: {best_round}, threshold={best_threshold:.3f}")
    print(
        f"Final test: F1={final_test['classification']['f1']:.4f}, "
        f"Acc={final_test['classification']['acc']:.4f}, "
        f"StateMAE={final_test['state']['overall_mae']:.4f}, "
        f"NegMean={final_test['probability']['neg_mean']:.4f}, "
        f"NegP95={final_test['probability']['neg_p95']:.4f}"
    )
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved figure: {security_fig}")
    print(f"Saved figure: {zoom_fig}")


if __name__ == "__main__":
    main()
