"""
Generate three local scatter-based boundary figures for case9 boundary-loop model.

Outputs:
  - figures/case9mod_boundaryloop_local_scatter_comp1.png
  - figures/case9mod_boundaryloop_local_scatter_comp2.png
  - figures/case9mod_boundaryloop_local_scatter_comp3.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_surrogate import build_case9mod_dataset  # noqa: E402


def _extract_component_centers(labels_2d: np.ndarray, p2_axis: np.ndarray, p3_axis: np.ndarray, top_k: int = 3) -> List[Tuple[float, float]]:
    ny, nx = labels_2d.shape
    mask = labels_2d > 0.5
    visited = np.zeros_like(mask, dtype=bool)
    nbr = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    comps: List[np.ndarray] = []

    for r in range(ny):
        for c in range(nx):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in nbr:
                    r2, c2 = rr + dr, cc + dc
                    if 0 <= r2 < ny and 0 <= c2 < nx and mask[r2, c2] and (not visited[r2, c2]):
                        visited[r2, c2] = True
                        stack.append((r2, c2))
            comp = np.zeros_like(mask, dtype=bool)
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
        use = edge if edge.any() else comp
        yy, xx = np.where(use)
        if yy.size == 0:
            continue
        cx = float(np.median(p2_axis[xx]))
        cy = float(np.median(p3_axis[yy]))
        centers.append((cx, cy))
    return centers


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate three local scatter boundary maps for case9")
    parser.add_argument("--data-dir", type=str, default=r"D:\安全域\1")
    parser.add_argument("--metrics", type=str, default=str(ROOT / "results" / "case9mod_boundaryloop_metrics.json"))
    parser.add_argument("--probs", type=str, default=str(ROOT / "results" / "case9mod_boundaryloop_probs.npy"))
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "figures"))
    parser.add_argument("--wx", type=float, default=38.0)
    parser.add_argument("--wy", type=float, default=32.0)
    args = parser.parse_args()

    ds = build_case9mod_dataset(Path(args.data_dir), seed=42, bg_multiplier=1000.0, n_guard_max=0)
    probs_full = np.load(args.probs)
    if len(probs_full) != len(ds.X_raw):
        raise ValueError("Probability size mismatch with dataset.")

    with open(args.metrics, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    threshold = float(metrics.get("best_threshold", 0.5))

    d2 = float(ds.p2_grid[1] - ds.p2_grid[0])
    d3 = float(ds.p3_grid[1] - ds.p3_grid[0])
    x_max = float(ds.p2_grid[-1])
    y_max = float(ds.p3_grid[-1])
    p2_axis = np.arange(0.0, x_max + 0.5 * d2, d2, dtype=np.float32)
    p3_axis = np.arange(0.0, y_max + 0.5 * d3, d3, dtype=np.float32)

    labels_2d = np.zeros((len(p3_axis), len(p2_axis)), dtype=np.float32)
    probs_2d = np.zeros((len(p3_axis), len(p2_axis)), dtype=np.float32)
    ix = np.rint((ds.X_raw[:, 0] - float(p2_axis[0])) / d2).astype(int)
    iy = np.rint((ds.X_raw[:, 1] - float(p3_axis[0])) / d3).astype(int)
    valid = (ix >= 0) & (ix < len(p2_axis)) & (iy >= 0) & (iy < len(p3_axis))
    labels_2d[iy[valid], ix[valid]] = ds.y_cls[valid].astype(np.float32)
    probs_2d[iy[valid], ix[valid]] = probs_full[valid].astype(np.float32)
    pred_2d = (probs_2d > threshold).astype(np.float32)

    centers = _extract_component_centers(labels_2d, p2_axis, p3_axis, top_k=3)
    if len(centers) < 3:
        raise RuntimeError(f"Only found {len(centers)} secure components; expected 3.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    for i, (cx, cy) in enumerate(centers[:3], start=1):
        x_keep = (p2_axis >= cx - float(args.wx)) & (p2_axis <= cx + float(args.wx))
        y_keep = (p3_axis >= cy - float(args.wy)) & (p3_axis <= cy + float(args.wy))
        xw = p2_axis[x_keep]
        yw = p3_axis[y_keep]
        XW, YW = np.meshgrid(xw, yw)

        lw = labels_2d[np.ix_(y_keep, x_keep)]
        pw = probs_2d[np.ix_(y_keep, x_keep)]
        prw = pred_2d[np.ix_(y_keep, x_keep)]

        xs = XW.ravel()
        ys = YW.ravel()
        ls = lw.ravel()
        ps = pw.ravel()

        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)

        def _draw_boundaries(ax):
            # Predicted boundary: white dashed with dark stroke for visibility.
            cs_pred = ax.contour(XW, YW, prw, levels=[0.5], colors="white", linewidths=1.6, linestyles="--")
            for coll in cs_pred.collections:
                coll.set_path_effects([pe.Stroke(linewidth=2.8, foreground="#222222"), pe.Normal()])

            # Traditional boundary: black solid.
            ax.contour(XW, YW, lw, levels=[0.5], colors="#111111", linewidths=1.6, linestyles="-")

        # Left: local label scatter + boundaries
        ax = axes[0]
        ax.scatter(xs[ls <= 0.5], ys[ls <= 0.5], s=10, c="#d62728", alpha=0.48, edgecolors="none", label="Infeasible")
        ax.scatter(xs[ls > 0.5], ys[ls > 0.5], s=12, c="#2ca02c", alpha=0.65, edgecolors="none", label="Feasible")
        _draw_boundaries(ax)

        fp = (prw > 0.5) & (lw <= 0.5)
        fn = (prw <= 0.5) & (lw > 0.5)
        if np.any(fp):
            ax.scatter(XW[fp], YW[fp], s=26, c="#ff8c00", marker="x", linewidths=1.1, label="FP")
        if np.any(fn):
            ax.scatter(XW[fn], YW[fn], s=26, c="#1f77b4", marker="+", linewidths=1.1, label="FN")

        ax.set_title(f"Local labels + boundaries (Component {i})")
        ax.set_xlabel("P_G2 (MW)")
        ax.set_ylabel("P_G3 (MW)")
        ax.grid(alpha=0.2)
        ax.plot([], [], color="white", linestyle="--", linewidth=1.6, label="Predicted boundary")
        ax.plot([], [], color="#111111", linestyle="-", linewidth=1.6, label="Traditional boundary")
        ax.legend(loc="upper right", fontsize=8)

        # Right: local probability scatter + boundaries
        ax = axes[1]
        sc = ax.scatter(xs, ys, c=ps, s=12, cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.90, edgecolors="none")
        _draw_boundaries(ax)
        if np.any(fp):
            ax.scatter(XW[fp], YW[fp], s=28, c="#ff8c00", marker="x", linewidths=1.1, label="FP")
        if np.any(fn):
            ax.scatter(XW[fn], YW[fn], s=28, c="#1f77b4", marker="+", linewidths=1.1, label="FN")
        ax.set_title(f"Local probability map (Component {i})")
        ax.set_xlabel("P_G2 (MW)")
        ax.set_ylabel("P_G3 (MW)")
        ax.grid(alpha=0.2)
        if np.any(fp) or np.any(fn):
            ax.legend(loc="upper right", fontsize=8)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("p_secure")

        out = out_dir / f"case9mod_boundaryloop_local_scatter_comp{i}.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
