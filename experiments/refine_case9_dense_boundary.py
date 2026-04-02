"""
Refine case9 boundary modeling with dense (0.02 MW) local sampling.

Workflow:
1) Use current boundary-loop model outputs to identify three secure components
   and FP/FN hard points.
2) Generate dense local points (0.02 MW) focused on each component interior,
   neighborhood, and boundary bands.
3) Query traditional AC-OPF oracle labels/states for dense points.
4) Augment training data and retrain EC-PDNet.
5) Export refined security-region figure and 3 local FP/FN scatter figures.

Outputs:
- results/case9mod_dense_refined_samples.csv
- results/case9mod_dense_refined_metrics.json
- results/case9mod_dense_refined_probs.npy
- results/case9mod_dense_refined_ecpd.pth
- figures/case9mod_boundary_dense_refined_security_region.png
- figures/case9mod_boundary_dense_refined_local_scatter_comp1.png
- figures/case9mod_boundary_dense_refined_local_scatter_comp2.png
- figures/case9mod_boundary_dense_refined_local_scatter_comp3.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_surrogate import (  # noqa: E402
    CASE9_STATE_COLUMNS,
    Case9Dataset,
    EnergyClosurePDNet,
    evaluate_energy_consistency,
    evaluate_full_state_model,
    make_dataloaders,
    split_indices,
    train_full_state_model,
    build_case9mod_dataset,
)

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")
logging.getLogger("pyomo").setLevel(logging.ERROR)
logging.getLogger("pyomo.core").setLevel(logging.ERROR)
logging.getLogger("pyomo.opt").setLevel(logging.ERROR)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_std(x: np.ndarray) -> np.ndarray:
    s = x.std(axis=0)
    s = np.where(s < 1e-6, 1.0, s)
    return s.astype(np.float32)


def _select_strict_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    fpr_target: float = 1e-3,
    t_min: float = 0.50,
    t_max: float = 0.95,
    n_grid: int = 91,
) -> Tuple[float, Dict[str, float]]:
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
        out = {
            "threshold": 0.6,
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


def _find_components(labels_2d: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> List[Dict]:
    ny, nx = labels_2d.shape
    mask = labels_2d > 0.5
    visited = np.zeros_like(mask, dtype=bool)
    nbr = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    comps: List[Dict] = []

    for r in range(ny):
        for c in range(nx):
            if (not mask[r, c]) or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            pts: List[Tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in nbr:
                    r2, c2 = rr + dr, cc + dc
                    if 0 <= r2 < ny and 0 <= c2 < nx and mask[r2, c2] and (not visited[r2, c2]):
                        visited[r2, c2] = True
                        stack.append((r2, c2))

            m = np.zeros_like(mask, dtype=bool)
            rr = np.array([p_[0] for p_ in pts], dtype=int)
            cc = np.array([p_[1] for p_ in pts], dtype=int)
            m[rr, cc] = True

            p = np.pad(m.astype(np.int16), 1, mode="constant", constant_values=0)
            n4 = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
            edge = m & (n4 < 4)
            yy_e, xx_e = np.where(edge)
            yy_i, xx_i = np.where(m)

            if yy_i.size == 0:
                continue
            centroid = (float(np.median(p2[xx_i])), float(np.median(p3[yy_i])))
            if yy_e.size > 0:
                bcenter = (float(np.median(p2[xx_e])), float(np.median(p3[yy_e])))
            else:
                bcenter = centroid

            comps.append({
                "mask": m,
                "edge": edge,
                "size": int(m.sum()),
                "centroid": centroid,
                "boundary_center": bcenter,
            })

    comps = sorted(comps, key=lambda d: d["size"], reverse=True)
    return comps[:3]


def _fallback_components_by_kmeans(labels_2d: np.ndarray, p2: np.ndarray, p3: np.ndarray, seed: int = 42) -> List[Dict]:
    yy, xx = np.where(labels_2d > 0.5)
    if yy.size < 3:
        return []
    pts = np.column_stack([p2[xx], p3[yy]])
    km = KMeans(n_clusters=3, random_state=seed, n_init="auto")
    lbl = km.fit_predict(pts)
    comps: List[Dict] = []
    for k in range(3):
        sub = pts[lbl == k]
        if len(sub) == 0:
            continue
        centroid = (float(np.median(sub[:, 0])), float(np.median(sub[:, 1])))
        # boundary center proxy: nearest point to cluster center
        c = km.cluster_centers_[k]
        d = np.sum((sub - c.reshape(1, -1)) ** 2, axis=1)
        bpt = sub[int(np.argmin(d))]
        comps.append(
            {
                "mask": np.zeros_like(labels_2d, dtype=bool),
                "edge": np.zeros_like(labels_2d, dtype=bool),
                "size": int(len(sub)),
                "centroid": centroid,
                "boundary_center": (float(bpt[0]), float(bpt[1])),
            }
        )
    comps = sorted(comps, key=lambda d: d["size"], reverse=True)
    return comps[:3]


def _regular_window(center: Tuple[float, float], half_w: float, step: float = 0.02) -> np.ndarray:
    cx, cy = center
    xs = np.arange(cx - half_w, cx + half_w + 0.5 * step, step, dtype=np.float64)
    ys = np.arange(cy - half_w, cy + half_w + 0.5 * step, step, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys)
    return np.column_stack([XX.ravel(), YY.ravel()])


def _quantize_and_clip(points: np.ndarray, p2_min: float, p2_max: float, p3_min: float, p3_max: float, step: float = 0.02) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    p[:, 0] = np.round(p[:, 0] / step) * step
    p[:, 1] = np.round(p[:, 1] / step) * step
    keep = (p[:, 0] >= p2_min) & (p[:, 0] <= p2_max) & (p[:, 1] >= p3_min) & (p[:, 1] <= p3_max)
    p = p[keep]
    key = np.round(p / step).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    p = p[np.sort(idx)]
    return p.astype(np.float32)


def _load_oracle_class(script_path: Path):
    spec = importlib.util.spec_from_file_location("case9_oracle_mod", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load oracle script: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ACOPFAnalyzer


def _query_oracle(points: np.ndarray, oracle_script: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Oracle = _load_oracle_class(oracle_script)
    cfg_path = ROOT / "results" / "ac_opf_dense_refined_config.json"
    if not cfg_path.exists():
        cfg = {
            "system": {
                "baseMVA": 100,
                "thermal_limit_factor": 1.0,
            },
            "calculation": {
                "n_points": 300,
                "batch_size": 100,
                "save_interval": 100,
            },
            "solver": {
                "max_iter": 100,
                "tol": 1e-4,
                "acceptable_tol": 1e-3,
                "print_level": 0,
            },
            "debug": {
                "enabled": False,
                "verbose": False,
                "save_infeasible": False,
            },
            "database": {
                "name": str(ROOT / "results" / "ac_opf_dense_refined.db"),
                "auto_clear": True,
            },
        }
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    analyzer = Oracle(config_file=str(cfg_path))
    n = len(points)

    y = np.zeros(n, dtype=np.float32)
    states = np.zeros((n, len(CASE9_STATE_COLUMNS)), dtype=np.float32)
    calc_time = np.zeros(n, dtype=np.float32)

    try:
        for i, (p2, p3) in enumerate(points):
            ok, sol, t = analyzer.solve_feasibility(float(p2) / 100.0, float(p3) / 100.0)
            calc_time[i] = float(t)
            if ok and sol is not None:
                y[i] = 1.0
                states[i, 0] = float(sol.get("p1_mw", 0.0))
                for k in range(1, 4):
                    states[i, k] = float(sol.get(f"q{k}_mvar", 0.0))
                for k in range(1, 10):
                    states[i, 3 + k] = float(sol.get(f"v{k}_pu", 1.0))
                for k in range(1, 10):
                    states[i, 12 + k] = float(sol.get(f"theta{k}_deg", 0.0))

            if (i + 1) % 200 == 0 or (i + 1) == n:
                print(f"Oracle progress: {i+1}/{n} | feasible={int(y[:i+1].sum())}")
    finally:
        analyzer.close()

    return y, states, calc_time


def _boundary_mask_knn(X: np.ndarray, y: np.ndarray, dist_th: float = 0.50) -> np.ndarray:
    out = np.zeros(len(X), dtype=np.float32)
    pos = y > 0.5
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return out

    tree_pos = KDTree(X[pos])
    tree_neg = KDTree(X[neg])

    d_pos, _ = tree_neg.query(X[pos], k=1)
    d_neg, _ = tree_pos.query(X[neg], k=1)
    out[pos] = (d_pos[:, 0] <= dist_th).astype(np.float32)
    out[neg] = (d_neg[:, 0] <= dist_th).astype(np.float32)
    return out


def _predict_probs(model: EnergyClosurePDNet, X_norm: np.ndarray, device: torch.device, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_norm), batch):
            xb = torch.from_numpy(X_norm[i : i + batch].astype(np.float32)).to(device)
            logits, _ = model(xb)
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out, axis=0)


def _plot_security_region(
    p2_axis: np.ndarray,
    p3_axis: np.ndarray,
    labels_2d: np.ndarray,
    probs_2d: np.ndarray,
    threshold: float,
    save_path: Path,
) -> None:
    XX, YY = np.meshgrid(p2_axis, p3_axis)
    pred_2d = (probs_2d > threshold).astype(np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)

    ax = axes[0]
    ax.contourf(XX, YY, labels_2d, levels=[-0.5, 0.5, 1.5], cmap=ListedColormap(["#f7b0b0", "#b6e3b6"]))
    ax.contour(XX, YY, labels_2d, levels=[0.5], colors="black", linewidths=1.2)
    ax.set_title("Traditional security region")
    ax.set_xlabel("P_G2 (MW)")
    ax.set_ylabel("P_G3 (MW)")
    ax.grid(alpha=0.2)

    ax = axes[1]
    c = ax.contourf(XX, YY, probs_2d, levels=np.linspace(0.0, 1.0, 21), cmap="viridis", vmin=0.0, vmax=1.0)
    # project convention: white dashed = predicted, black solid = traditional
    cs = ax.contour(XX, YY, pred_2d, levels=[0.5], colors="white", linewidths=1.4, linestyles="--")
    for coll in cs.collections:
        coll.set_path_effects([pe.Stroke(linewidth=2.6, foreground="#222222"), pe.Normal()])
    ax.contour(XX, YY, labels_2d, levels=[0.5], colors="black", linewidths=1.2)
    ax.set_title(f"Dense-refined model probability (th={threshold:.3f})")
    ax.set_xlabel("P_G2 (MW)")
    ax.set_ylabel("P_G3 (MW)")
    ax.grid(alpha=0.2)
    ax.plot([], [], color="white", linestyle="--", linewidth=1.4, label="Predicted boundary")
    ax.plot([], [], color="black", linestyle="-", linewidth=1.2, label="Traditional boundary")
    ax.legend(loc="upper right", fontsize=8)
    cb = fig.colorbar(c, ax=ax)
    cb.set_label("p_secure")

    save_path.parent.mkdir(exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def _plot_local_scatter(
    points: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    title: str,
    save_path: Path,
) -> None:
    x = points[:, 0]
    y = points[:, 1]
    pred = (probs > threshold).astype(np.float32)
    fp = (pred > 0.5) & (y_true <= 0.5)
    fn = (pred <= 0.5) & (y_true > 0.5)

    x_unique = np.unique(np.round(x, 2))
    y_unique = np.unique(np.round(y, 2))
    XW, YW = np.meshgrid(x_unique, y_unique)
    lw = np.full_like(XW, np.nan, dtype=np.float32)
    prw = np.full_like(XW, np.nan, dtype=np.float32)

    x0 = float(x_unique[0])
    y0 = float(y_unique[0])
    step_x = float(x_unique[1] - x_unique[0]) if len(x_unique) > 1 else 0.02
    step_y = float(y_unique[1] - y_unique[0]) if len(y_unique) > 1 else 0.02
    ix = np.rint((x - x0) / step_x).astype(int)
    iy = np.rint((y - y0) / step_y).astype(int)
    valid = (ix >= 0) & (ix < len(x_unique)) & (iy >= 0) & (iy < len(y_unique))
    lw[iy[valid], ix[valid]] = y_true[valid]
    prw[iy[valid], ix[valid]] = pred[valid]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)

    def _draw_boundaries(ax):
        m1 = np.nan_to_num(prw, nan=0.0)
        m2 = np.nan_to_num(lw, nan=0.0)
        cs = ax.contour(XW, YW, m1, levels=[0.5], colors="white", linewidths=1.6, linestyles="--")
        for coll in cs.collections:
            coll.set_path_effects([pe.Stroke(linewidth=2.8, foreground="#222222"), pe.Normal()])
        ax.contour(XW, YW, m2, levels=[0.5], colors="black", linewidths=1.6)

    ax = axes[0]
    ax.scatter(x[y_true <= 0.5], y[y_true <= 0.5], s=10, c="#d62728", alpha=0.45, edgecolors="none", label="Infeasible")
    ax.scatter(x[y_true > 0.5], y[y_true > 0.5], s=10, c="#2ca02c", alpha=0.60, edgecolors="none", label="Feasible")
    _draw_boundaries(ax)
    if np.any(fp):
        ax.scatter(x[fp], y[fp], s=28, c="#ff8c00", marker="x", linewidths=1.1, label="FP")
    if np.any(fn):
        ax.scatter(x[fn], y[fn], s=28, c="#1f77b4", marker="+", linewidths=1.1, label="FN")
    ax.plot([], [], color="white", linestyle="--", linewidth=1.6, label="Predicted boundary")
    ax.plot([], [], color="black", linestyle="-", linewidth=1.6, label="Traditional boundary")
    ax.set_title(f"{title}: labels + boundaries")
    ax.set_xlabel("P_G2 (MW)")
    ax.set_ylabel("P_G3 (MW)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    sc = ax.scatter(x, y, c=probs, s=10, cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.90, edgecolors="none")
    _draw_boundaries(ax)
    if np.any(fp):
        ax.scatter(x[fp], y[fp], s=28, c="#ff8c00", marker="x", linewidths=1.1, label="FP")
    if np.any(fn):
        ax.scatter(x[fn], y[fn], s=28, c="#1f77b4", marker="+", linewidths=1.1, label="FN")
    ax.set_title(f"{title}: probability")
    ax.set_xlabel("P_G2 (MW)")
    ax.set_ylabel("P_G3 (MW)")
    ax.grid(alpha=0.2)
    if np.any(fp) or np.any(fn):
        ax.legend(loc="upper right", fontsize=8)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("p_secure")

    save_path.parent.mkdir(exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense case9 boundary refinement with FP/FN focus")
    parser.add_argument("--data-dir", type=str, default=r"D:\安全域\1")
    parser.add_argument("--oracle-script", type=str, default=r"D:\安全域\1\case9线路热极限.py")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])

    parser.add_argument("--dense-step", type=float, default=0.02)
    parser.add_argument("--half-interior", type=float, default=0.60)
    parser.add_argument("--half-boundary", type=float, default=0.80)
    parser.add_argument("--fpfn-seeds-per-comp", type=int, default=36)
    parser.add_argument("--fpfn-half", type=float, default=0.04)

    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--lambda-state", type=float, default=2.0)
    parser.add_argument("--lambda-voltage", type=float, default=0.05)
    parser.add_argument("--lambda-polar", type=float, default=0.28)
    parser.add_argument("--lambda-hard-neg", type=float, default=0.90)
    parser.add_argument("--hard-neg-th", type=float, default=0.12)
    parser.add_argument("--lambda-hard-pos", type=float, default=0.08)
    parser.add_argument("--hard-pos-floor", type=float, default=0.82)
    parser.add_argument("--boundary-weight", type=float, default=3.0)
    parser.add_argument("--boundary-hard-neg-boost", type=float, default=4.0)
    parser.add_argument("--p1-weight", type=float, default=4.0)
    parser.add_argument("--strict-fpr-target", type=float, default=8e-4)
    args = parser.parse_args()

    _set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    ds_visual = build_case9mod_dataset(data_dir, seed=args.seed, bg_multiplier=1000.0, n_guard_max=0)
    ds_base = build_case9mod_dataset(data_dir, seed=args.seed, bg_multiplier=8.0, n_guard_max=3000)
    p2 = ds_visual.p2_grid
    p3 = ds_visual.p3_grid
    labels_2d = ds_visual.y_cls.reshape(len(p3), len(p2))

    # Existing model outputs (for FP/FN-focused densification)
    old_probs_path = ROOT / "results" / "case9mod_boundaryloop_probs.npy"
    old_metrics_path = ROOT / "results" / "case9mod_boundaryloop_metrics.json"
    old_probs = np.load(old_probs_path)
    with open(old_metrics_path, "r", encoding="utf-8") as f:
        old_metrics = json.load(f)
    old_th = float(old_metrics.get("best_threshold", 0.645))
    pred_old = (old_probs > old_th).astype(np.float32)
    pred_old_2d = pred_old.reshape(len(p3), len(p2))
    fp_2d = (pred_old_2d > 0.5) & (labels_2d <= 0.5)
    fn_2d = (pred_old_2d <= 0.5) & (labels_2d > 0.5)

    comps = _find_components(labels_2d, p2, p3)
    if len(comps) < 3:
        print(f"Warning: connected-component split found {len(comps)} region(s); fallback to k-means mode split.")
        comps = _fallback_components_by_kmeans(labels_2d, p2, p3, seed=args.seed)
    if len(comps) < 3:
        raise RuntimeError(f"Expected 3 security subregions after fallback, got {len(comps)}")

    rng = np.random.default_rng(args.seed)
    dense_points_parts: List[np.ndarray] = []
    eval_windows: List[np.ndarray] = []
    comp_centers: List[Tuple[float, float]] = []

    fpfn_offsets = np.arange(-float(args.fpfn_half), float(args.fpfn_half) + 1e-9, float(args.dense_step), dtype=np.float64)

    for ci, comp in enumerate(comps, start=1):
        c_in = comp["centroid"]
        c_bd = comp["boundary_center"]
        comp_centers.append(c_bd)

        win_in = _regular_window(c_in, half_w=float(args.half_interior), step=float(args.dense_step))
        win_bd = _regular_window(c_bd, half_w=float(args.half_boundary), step=float(args.dense_step))
        eval_windows.append(win_bd)
        dense_points_parts.extend([win_in, win_bd])

        # FP/FN seed patches from old model around this component
        mask_near = comp["mask"]
        if not np.any(mask_near):
            # Fallback path (k-means components): use distance-to-center neighborhood.
            XX, YY = np.meshgrid(p2, p3)
            cx, cy = c_bd
            mask_near = ((XX - cx) ** 2 + (YY - cy) ** 2) <= (2.0 ** 2)

        fp_idx = np.argwhere(fp_2d & mask_near)
        fn_idx = np.argwhere(fn_2d & mask_near)
        if len(fp_idx) > args.fpfn_seeds_per_comp:
            fp_idx = fp_idx[rng.choice(len(fp_idx), size=args.fpfn_seeds_per_comp, replace=False)]
        if len(fn_idx) > args.fpfn_seeds_per_comp:
            fn_idx = fn_idx[rng.choice(len(fn_idx), size=args.fpfn_seeds_per_comp, replace=False)]

        for idx_arr in [fp_idx, fn_idx]:
            for iy, ix in idx_arr:
                x0 = float(p2[ix])
                y0 = float(p3[iy])
                XX, YY = np.meshgrid(x0 + fpfn_offsets, y0 + fpfn_offsets)
                dense_points_parts.append(np.column_stack([XX.ravel(), YY.ravel()]))

    dense_points = np.vstack(dense_points_parts)
    dense_points = _quantize_and_clip(
        dense_points,
        p2_min=float(p2[0]),
        p2_max=float(p2[-1]),
        p3_min=float(p3[0]),
        p3_max=float(p3[-1]),
        step=float(args.dense_step),
    )
    print(f"Dense candidate points: {len(dense_points)}")

    y_dense, state_dense, t_dense = _query_oracle(dense_points, Path(args.oracle_script))
    print(f"Dense oracle feasible: {int(y_dense.sum())}/{len(y_dense)}")

    # Save supplemental dense sample table
    dense_df = pd.DataFrame({
        "p2_mw": dense_points[:, 0],
        "p3_mw": dense_points[:, 1],
        "is_feasible": y_dense.astype(int),
        "calculation_time": t_dense,
    })
    for j, name in enumerate(CASE9_STATE_COLUMNS):
        col = state_dense[:, j].copy()
        col[y_dense < 0.5] = np.nan
        dense_df[name] = col
    dense_csv = ROOT / "results" / "case9mod_dense_refined_samples.csv"
    dense_df.to_csv(dense_csv, index=False, encoding="utf-8-sig")

    # Build augmented training dataset: full coarse + dense supplement
    X_raw = np.vstack([ds_base.X_raw, dense_points]).astype(np.float32)
    y_cls = np.concatenate([ds_base.y_cls, y_dense]).astype(np.float32)
    y_state_raw = np.vstack([ds_base.y_state_raw, state_dense]).astype(np.float32)
    state_mask = np.concatenate([ds_base.state_mask, y_dense]).astype(np.float32)

    boundary_mask = _boundary_mask_knn(X_raw, y_cls, dist_th=0.50)

    x_mean = X_raw.mean(axis=0).astype(np.float32)
    x_std = _safe_std(X_raw)

    feas = state_mask > 0.5
    state_mean = y_state_raw[feas].mean(axis=0).astype(np.float32)
    state_std = _safe_std(y_state_raw[feas])
    y_state_norm = ((y_state_raw - state_mean.reshape(1, -1)) / state_std.reshape(1, -1)).astype(np.float32)

    ds_aug = Case9Dataset(
        X_raw=X_raw,
        X_norm=((X_raw - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)).astype(np.float32),
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
        p2_grid=ds_visual.p2_grid,
        p3_grid=ds_visual.p3_grid,
        total_load_mw=ds_visual.total_load_mw,
        secure_lookup=ds_visual.secure_lookup,
        secure_points=X_raw[y_cls > 0.5],
        insecure_points=X_raw[y_cls <= 0.5],
    )

    split = split_indices(ds_aug.y_cls, seed=args.seed)
    loaders = make_dataloaders(ds_aug, split, batch_size=args.batch_size)

    model = EnergyClosurePDNet(
        input_dim=2,
        n_state=len(CASE9_STATE_COLUMNS),
        x_mean=ds_aug.x_mean,
        x_std=ds_aug.x_std,
        state_mean=ds_aug.state_mean,
        state_std=ds_aug.state_std,
        total_load_mw=ds_aug.total_load_mw,
        dropout=0.1,
    )

    state_w = np.ones(len(CASE9_STATE_COLUMNS), dtype=np.float32)
    state_w[0] = float(args.p1_weight)

    hist = train_full_state_model(
        model=model,
        loaders=loaders,
        dataset=ds_aug,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        lambda_state=args.lambda_state,
        lambda_voltage=args.lambda_voltage,
        lambda_monotonic=0.0,
        lambda_polar=args.lambda_polar,
        lambda_hard_neg=args.lambda_hard_neg,
        hard_neg_th=args.hard_neg_th,
        lambda_hard_pos=args.lambda_hard_pos,
        hard_pos_floor=args.hard_pos_floor,
        boundary_weight=args.boundary_weight,
        boundary_hard_neg_boost=args.boundary_hard_neg_boost,
        state_weight=state_w,
        patience=35,
    )

    # Strict threshold on validation
    val_ids = split["val"]
    val_probs = _predict_probs(model, ds_aug.X_norm[val_ids], device=device)
    val_labels = ds_aug.y_cls[val_ids]
    th, th_info = _select_strict_threshold(
        val_probs,
        val_labels,
        fpr_target=float(args.strict_fpr_target),
        t_min=0.50,
        t_max=0.95,
        n_grid=91,
    )

    test_eval = evaluate_full_state_model(model, loaders["test"], ds_aug, device, threshold=th)
    closure_eval = evaluate_energy_consistency(model, loaders["test"], ds_aug, device)

    # Predict on full coarse lattice for global region figure
    X_full_norm = ((ds_visual.X_raw - ds_aug.x_mean.reshape(1, -1)) / ds_aug.x_std.reshape(1, -1)).astype(np.float32)
    probs_full = _predict_probs(model, X_full_norm, device=device)
    probs_full_2d = probs_full.reshape(len(p3), len(p2))

    pred_full = (probs_full > th).astype(np.float32)
    y_full = ds_visual.y_cls.astype(np.float32)
    tp = int(np.sum((pred_full > 0.5) & (y_full > 0.5)))
    fp = int(np.sum((pred_full > 0.5) & (y_full <= 0.5)))
    fn = int(np.sum((pred_full <= 0.5) & (y_full > 0.5)))
    tn = int(np.sum((pred_full <= 0.5) & (y_full <= 0.5)))
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1_full = 2 * prec * rec / (prec + rec + 1e-12)
    acc_full = float((pred_full == y_full).mean())

    # Save checkpoint and metrics
    ckpt = ROOT / "results" / "case9mod_dense_refined_ecpd.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_mean": ds_aug.x_mean,
            "x_std": ds_aug.x_std,
            "state_mean": ds_aug.state_mean,
            "state_std": ds_aug.state_std,
            "state_names": ds_aug.state_names,
            "best_threshold": float(th),
        },
        ckpt,
    )
    probs_path = ROOT / "results" / "case9mod_dense_refined_probs.npy"
    np.save(probs_path, probs_full.astype(np.float32))

    fig_global = ROOT / "figures" / "case9mod_boundary_dense_refined_security_region.png"
    _plot_security_region(
        p2_axis=p2,
        p3_axis=p3,
        labels_2d=labels_2d,
        probs_2d=probs_full_2d,
        threshold=th,
        save_path=fig_global,
    )

    # Local dense scatter figures for 3 components
    # Build oracle map for fast lookup on dense-sampled points
    key_dense = np.round(dense_points / float(args.dense_step)).astype(np.int64)
    dense_key_to_idx = {tuple(k.tolist()): i for i, k in enumerate(key_dense)}

    local_figs = []
    for i, center in enumerate(comp_centers, start=1):
        eval_pts = _regular_window(center, half_w=float(args.half_boundary), step=float(args.dense_step))
        eval_pts = _quantize_and_clip(
            eval_pts,
            float(p2[0]),
            float(p2[-1]),
            float(p3[0]),
            float(p3[-1]),
            step=float(args.dense_step),
        )
        k_eval = np.round(eval_pts / float(args.dense_step)).astype(np.int64)

        idx_list = []
        missing = []
        for k in k_eval:
            idx = dense_key_to_idx.get(tuple(k.tolist()))
            if idx is None:
                missing.append(k)
            else:
                idx_list.append(idx)

        if missing:
            # Safety fallback: query oracle for missing points (normally none).
            add_pts = np.asarray(missing, dtype=np.float32) * float(args.dense_step)
            y_add, state_add, _ = _query_oracle(add_pts, Path(args.oracle_script))
            start = len(dense_points)
            dense_points = np.vstack([dense_points, add_pts]).astype(np.float32)
            y_dense = np.concatenate([y_dense, y_add]).astype(np.float32)
            state_dense = np.vstack([state_dense, state_add]).astype(np.float32)
            for j, kk in enumerate(np.round(add_pts / float(args.dense_step)).astype(np.int64)):
                dense_key_to_idx[tuple(kk.tolist())] = start + j
                idx_list.append(start + j)

        idx_arr = np.array(idx_list, dtype=int)
        pts_i = dense_points[idx_arr]
        y_i = y_dense[idx_arr]
        X_i_norm = ((pts_i - ds_aug.x_mean.reshape(1, -1)) / ds_aug.x_std.reshape(1, -1)).astype(np.float32)
        probs_i = _predict_probs(model, X_i_norm, device=device)

        fig_i = ROOT / "figures" / f"case9mod_boundary_dense_refined_local_scatter_comp{i}.png"
        _plot_local_scatter(
            points=pts_i,
            y_true=y_i,
            probs=probs_i,
            threshold=float(th),
            title=f"Dense Local Component {i}",
            save_path=fig_i,
        )
        local_figs.append(str(fig_i))

    metrics = {
        "method": "EC-PDNet dense refined (FP/FN-aware)",
        "dense_sampling": {
            "step_mw": float(args.dense_step),
            "half_interior": float(args.half_interior),
            "half_boundary": float(args.half_boundary),
            "fpfn_seeds_per_comp": int(args.fpfn_seeds_per_comp),
            "fpfn_half": float(args.fpfn_half),
            "dense_points_total": int(len(dense_points)),
            "dense_feasible": int(y_dense.sum()),
        },
        "threshold": {
            "strict": float(th),
            "selection": th_info,
        },
        "coarse_fullgrid": {
            "acc": float(acc_full),
            "f1": float(f1_full),
            "precision": float(prec),
            "recall": float(rec),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "tn": int(tn),
        },
        "test_eval_augmented": test_eval,
        "test_closure_augmented": closure_eval,
        "artifacts": {
            "dense_csv": str(dense_csv),
            "checkpoint": str(ckpt),
            "probs_full": str(probs_path),
            "figure_security": str(fig_global),
            "figures_local": local_figs,
        },
        "train_history_tail": {
            "train_total": hist.get("train_total", [])[-5:],
            "val_total": hist.get("val_total", [])[-5:],
            "val_f1@0.5": hist.get("val_f1@0.5", [])[-5:],
            "val_state_mae": hist.get("val_state_mae", [])[-5:],
        },
    }

    metrics_path = ROOT / "results" / "case9mod_dense_refined_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\nDense refinement complete.")
    print(f"Strict threshold: {th:.4f}")
    print(f"Full-grid acc={acc_full:.4f}, f1={f1_full:.4f}, fp={fp}, fn={fn}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved security fig: {fig_global}")
    for p_ in local_figs:
        print(f"Saved local fig: {p_}")


if __name__ == "__main__":
    main()
