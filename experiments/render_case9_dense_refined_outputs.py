"""
Render optimized dense-refined case9 figures from an existing trained checkpoint.

This does NOT rerun oracle solving/training. It only re-renders figures with a
chosen threshold to control FP/FN tradeoff.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_surrogate import EnergyClosurePDNet, build_case9mod_dataset  # noqa: E402


def _predict_probs(model: EnergyClosurePDNet, X_norm: np.ndarray, device: str = "cpu", batch: int = 4096) -> np.ndarray:
    import torch

    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_norm), batch):
            xb = torch.from_numpy(X_norm[i : i + batch].astype(np.float32)).to(dev)
            logits, _ = model(xb)
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out, axis=0)


def _kmeans_centers(labels_2d: np.ndarray, p2: np.ndarray, p3: np.ndarray, seed: int = 42) -> List[Tuple[float, float]]:
    yy, xx = np.where(labels_2d > 0.5)
    pts = np.column_stack([p2[xx], p3[yy]])
    km = KMeans(n_clusters=3, random_state=seed, n_init="auto")
    km.fit(pts)
    centers = [(float(c[0]), float(c[1])) for c in km.cluster_centers_]
    return centers


def _plot_security_region(p2: np.ndarray, p3: np.ndarray, labels_2d: np.ndarray, probs_2d: np.ndarray, threshold: float, save: Path) -> None:
    XX, YY = np.meshgrid(p2, p3)
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
    cs = ax.contour(XX, YY, pred_2d, levels=[0.5], colors="white", linewidths=1.6, linestyles="--")
    for coll in cs.collections:
        coll.set_path_effects([pe.Stroke(linewidth=2.8, foreground="#222222"), pe.Normal()])
    ax.contour(XX, YY, labels_2d, levels=[0.5], colors="black", linewidths=1.6)
    ax.plot([], [], color="white", linestyle="--", linewidth=1.6, label="Predicted boundary")
    ax.plot([], [], color="black", linestyle="-", linewidth=1.6, label="Traditional boundary")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Dense-refined map (th={threshold:.3f})")
    ax.set_xlabel("P_G2 (MW)")
    ax.set_ylabel("P_G3 (MW)")
    ax.grid(alpha=0.2)
    fig.colorbar(c, ax=ax, label="p_secure")

    save.parent.mkdir(exist_ok=True)
    fig.savefig(save, dpi=300)
    plt.close(fig)


def _plot_local(points: np.ndarray, y_true: np.ndarray, probs: np.ndarray, th: float, save: Path, title: str) -> None:
    x = points[:, 0]
    y = points[:, 1]
    pred = (probs > th).astype(np.float32)
    fp = (pred > 0.5) & (y_true <= 0.5)
    fn = (pred <= 0.5) & (y_true > 0.5)

    x_u = np.unique(np.round(x, 2))
    y_u = np.unique(np.round(y, 2))
    XW, YW = np.meshgrid(x_u, y_u)
    lw = np.full_like(XW, np.nan, dtype=np.float32)
    pw = np.full_like(XW, np.nan, dtype=np.float32)
    x0 = float(x_u[0])
    y0 = float(y_u[0])
    dx = float(x_u[1] - x_u[0]) if len(x_u) > 1 else 0.02
    dy = float(y_u[1] - y_u[0]) if len(y_u) > 1 else 0.02
    ix = np.rint((x - x0) / dx).astype(int)
    iy = np.rint((y - y0) / dy).astype(int)
    v = (ix >= 0) & (ix < len(x_u)) & (iy >= 0) & (iy < len(y_u))
    lw[iy[v], ix[v]] = y_true[v]
    pw[iy[v], ix[v]] = pred[v]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)

    def draw_bound(ax):
        m1 = np.nan_to_num(pw, nan=0.0)
        m2 = np.nan_to_num(lw, nan=0.0)
        cs = ax.contour(XW, YW, m1, levels=[0.5], colors="white", linewidths=1.6, linestyles="--")
        for coll in cs.collections:
            coll.set_path_effects([pe.Stroke(linewidth=2.8, foreground="#222222"), pe.Normal()])
        ax.contour(XW, YW, m2, levels=[0.5], colors="black", linewidths=1.6)

    ax = axes[0]
    ax.scatter(x[y_true <= 0.5], y[y_true <= 0.5], s=10, c="#d62728", alpha=0.45, edgecolors="none", label="Infeasible")
    ax.scatter(x[y_true > 0.5], y[y_true > 0.5], s=10, c="#2ca02c", alpha=0.6, edgecolors="none", label="Feasible")
    draw_bound(ax)
    if np.any(fp):
        ax.scatter(x[fp], y[fp], s=28, c="#ff8c00", marker="x", linewidths=1.1, label="FP")
    if np.any(fn):
        ax.scatter(x[fn], y[fn], s=28, c="#1f77b4", marker="+", linewidths=1.1, label="FN")
    ax.plot([], [], color="white", linestyle="--", linewidth=1.6, label="Predicted boundary")
    ax.plot([], [], color="black", linestyle="-", linewidth=1.6, label="Traditional boundary")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{title}: labels + boundaries")
    ax.set_xlabel("P_G2 (MW)")
    ax.set_ylabel("P_G3 (MW)")
    ax.grid(alpha=0.2)

    ax = axes[1]
    sc = ax.scatter(x, y, c=probs, s=10, cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.9, edgecolors="none")
    draw_bound(ax)
    if np.any(fp):
        ax.scatter(x[fp], y[fp], s=28, c="#ff8c00", marker="x", linewidths=1.1, label="FP")
    if np.any(fn):
        ax.scatter(x[fn], y[fn], s=28, c="#1f77b4", marker="+", linewidths=1.1, label="FN")
    if np.any(fp) or np.any(fn):
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{title}: probability")
    ax.set_xlabel("P_G2 (MW)")
    ax.set_ylabel("P_G3 (MW)")
    ax.grid(alpha=0.2)
    fig.colorbar(sc, ax=ax, label="p_secure")

    save.parent.mkdir(exist_ok=True)
    fig.savefig(save, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render optimized dense refined figures")
    parser.add_argument("--data-dir", type=str, default=r"D:\安全域\1")
    parser.add_argument("--dense-csv", type=str, default=str(ROOT / "results" / "case9mod_dense_refined_samples.csv"))
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "results" / "case9mod_dense_refined_ecpd.pth"))
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--window-half", type=float, default=0.80)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    import torch

    ds = build_case9mod_dataset(Path(args.data_dir), seed=42, bg_multiplier=1000.0, n_guard_max=0)
    p2, p3 = ds.p2_grid, ds.p3_grid
    labels_2d = ds.y_cls.reshape(len(p3), len(p2))

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = EnergyClosurePDNet(
        input_dim=2,
        n_state=len(ck["state_names"]),
        x_mean=np.asarray(ck["x_mean"], dtype=np.float32),
        x_std=np.asarray(ck["x_std"], dtype=np.float32),
        state_mean=np.asarray(ck["state_mean"], dtype=np.float32),
        state_std=np.asarray(ck["state_std"], dtype=np.float32),
        total_load_mw=float(ds.total_load_mw),
        dropout=0.1,
    )
    model.load_state_dict(ck["state_dict"], strict=True)

    X_full_norm = ((ds.X_raw - np.asarray(ck["x_mean"], dtype=np.float32).reshape(1, -1)) / np.asarray(ck["x_std"], dtype=np.float32).reshape(1, -1)).astype(np.float32)
    probs_full = _predict_probs(model, X_full_norm, device=args.device)
    probs_2d = probs_full.reshape(len(p3), len(p2))

    fig_global = ROOT / "figures" / "case9mod_boundary_dense_refined_security_region_opt.png"
    _plot_security_region(p2, p3, labels_2d, probs_2d, float(args.threshold), fig_global)

    dense_df = pd.read_csv(args.dense_csv)
    dense_pts = dense_df[["p2_mw", "p3_mw"]].to_numpy(dtype=np.float32)
    y_true = dense_df["is_feasible"].to_numpy(dtype=np.float32)
    X_dense_norm = ((dense_pts - np.asarray(ck["x_mean"], dtype=np.float32).reshape(1, -1)) / np.asarray(ck["x_std"], dtype=np.float32).reshape(1, -1)).astype(np.float32)
    probs_dense = _predict_probs(model, X_dense_norm, device=args.device)

    centers = _kmeans_centers(labels_2d, p2, p3, seed=42)
    local_paths = []
    for i, (cx, cy) in enumerate(centers, start=1):
        m = (np.abs(dense_pts[:, 0] - cx) <= float(args.window_half)) & (np.abs(dense_pts[:, 1] - cy) <= float(args.window_half))
        pts_i = dense_pts[m]
        y_i = y_true[m]
        p_i = probs_dense[m]
        out = ROOT / "figures" / f"case9mod_boundary_dense_refined_local_scatter_comp{i}_opt.png"
        _plot_local(pts_i, y_i, p_i, float(args.threshold), out, f"Dense Local Component {i}")
        local_paths.append(str(out))

    pred = (probs_full > float(args.threshold)).astype(np.float32)
    y = ds.y_cls.astype(np.float32)
    tp = int(np.sum((pred > 0.5) & (y > 0.5)))
    fp = int(np.sum((pred > 0.5) & (y <= 0.5)))
    fn = int(np.sum((pred <= 0.5) & (y > 0.5)))
    tn = int(np.sum((pred <= 0.5) & (y <= 0.5)))
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    acc = float((pred == y).mean())

    out_json = ROOT / "results" / "case9mod_dense_refined_metrics_opt.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": float(args.threshold),
                "full_grid": {
                    "acc": acc,
                    "f1": float(f1),
                    "precision": float(prec),
                    "recall": float(rec),
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "tn": tn,
                },
                "artifacts": {
                    "figure_security": str(fig_global),
                    "figures_local": local_paths,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved: {fig_global}")
    for pth in local_paths:
        print(f"Saved: {pth}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
