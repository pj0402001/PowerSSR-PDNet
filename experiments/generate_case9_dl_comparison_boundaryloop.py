"""
Generate case9mod DL-comparison figure using boundary-loop model outputs.

This script writes:
  - figures/case9mod_dl_comparison.png

It keeps the publication-like 3-panel layout and compares against traditional
labels while explicitly showing the [0,10) MW guard-band infeasible region.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_surrogate import build_case9mod_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate case9mod DL comparison from boundary-loop outputs")
    parser.add_argument("--data-dir", type=str, default=r"D:\安全域\1")
    parser.add_argument(
        "--metrics", type=str, default=str(ROOT / "results" / "case9mod_boundaryloop_metrics.json")
    )
    parser.add_argument(
        "--probs", type=str, default=str(ROOT / "results" / "case9mod_boundaryloop_probs.npy")
    )
    parser.add_argument(
        "--output", type=str, default=str(ROOT / "figures" / "case9mod_dl_comparison.png")
    )
    args = parser.parse_args()

    ds = build_case9mod_dataset(Path(args.data_dir), seed=42, bg_multiplier=1000.0, n_guard_max=0)
    probs_full = np.load(args.probs)
    if len(probs_full) != len(ds.X_raw):
        raise ValueError(f"Probability length mismatch: got {len(probs_full)}, expected {len(ds.X_raw)}")

    with open(args.metrics, "r", encoding="utf-8") as f:
        m = json.load(f)
    th = float(m.get("best_threshold", 0.5))

    d2 = float(ds.p2_grid[1] - ds.p2_grid[0])
    d3 = float(ds.p3_grid[1] - ds.p3_grid[0])
    x_max = float(ds.p2_grid[-1])
    y_max = float(ds.p3_grid[-1])
    x_arr = np.arange(0.0, x_max + 0.5 * d2, d2, dtype=np.float32)
    y_arr = np.arange(0.0, y_max + 0.5 * d3, d3, dtype=np.float32)
    XX, YY = np.meshgrid(x_arr, y_arr)

    labels_2d = np.zeros((len(y_arr), len(x_arr)), dtype=np.float32)
    probs_2d = np.zeros((len(y_arr), len(x_arr)), dtype=np.float32)

    ix = np.rint((ds.X_raw[:, 0] - float(x_arr[0])) / d2).astype(int)
    iy = np.rint((ds.X_raw[:, 1] - float(y_arr[0])) / d3).astype(int)
    valid = (ix >= 0) & (ix < len(x_arr)) & (iy >= 0) & (iy < len(y_arr))
    labels_2d[iy[valid], ix[valid]] = ds.y_cls[valid].astype(np.float32)
    probs_2d[iy[valid], ix[valid]] = probs_full[valid].astype(np.float32)

    pred_2d = (probs_2d > th).astype(np.float32)
    grid_acc = float((pred_2d == labels_2d).mean())

    secure_cmap = LinearSegmentedColormap.from_list("security", ["#f4efe5", "#d8e7d2", "#2d6a4f"])
    truth_colors = ["#efe7da", "#5b8c5a"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.1))
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor("#fbfaf7")
        ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.6)
        ax.axvspan(0.0, 10.0, color="#d9d9d9", alpha=0.35, zorder=0)
        ax.axhspan(0.0, 10.0, color="#d9d9d9", alpha=0.35, zorder=0)
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(0.0, y_max)
        ax.set_xlabel("P_G2 (MW)", fontweight="bold")
        ax.set_ylabel("P_G3 (MW)", fontweight="bold")

    # (a) Traditional reference
    ax = axes[0]
    ax.contourf(XX, YY, labels_2d, levels=[-0.5, 0.5, 1.5], colors=truth_colors, alpha=0.92)
    if labels_2d.max() > 0.5 and labels_2d.min() < 0.5:
        ax.contour(XX, YY, labels_2d, levels=[0.5], colors="#24323b", linewidths=1.9)
    ax.set_title("Traditional IPOPT reference", fontweight="bold")
    ax.text(0.02, 0.98, "a", transform=ax.transAxes, ha="left", va="top", fontweight="bold")

    # (b) Boundary-loop probability
    ax = axes[1]
    cs = ax.contourf(XX, YY, probs_2d, levels=np.linspace(0, 1, 21), cmap=secure_cmap, vmin=0.0, vmax=1.0)
    plt.colorbar(cs, ax=ax, label="Security score", fraction=0.045, pad=0.02)
    if probs_2d.max() > th and probs_2d.min() < th:
        ax.contour(XX, YY, probs_2d, levels=[th], colors="#111111", linewidths=2.2)
    if labels_2d.max() > 0.5 and labels_2d.min() < 0.5:
        ax.contour(XX, YY, labels_2d, levels=[0.5], colors="white", linewidths=1.4, linestyles="--")
    ax.set_title(f"EC-PDNet+WLDG-BE (grid acc.={grid_acc:.3f}, th={th:.2f})", fontweight="bold")
    ax.plot([], [], color="#111111", lw=2.2, label="Predicted boundary")
    ax.plot([], [], color="white", lw=1.4, linestyle="--", label="True boundary")
    ax.legend(loc="upper right", fontsize=8)
    ax.text(0.02, 0.98, "b", transform=ax.transAxes, ha="left", va="top", fontweight="bold")

    # (c) Disagreement map
    ax = axes[2]
    err = np.abs(pred_2d - labels_2d)
    ax.contourf(XX, YY, labels_2d, levels=[-0.5, 0.5, 1.5], colors=truth_colors, alpha=0.35)
    if err.any():
        ax.contourf(XX, YY, err, levels=[0.5, 1.5], colors=["#8c2f39"], alpha=0.85)
    if labels_2d.max() > 0.5 and labels_2d.min() < 0.5:
        ax.contour(XX, YY, labels_2d, levels=[0.5], colors="#24323b", linewidths=1.5)
    ax.set_title("Prediction disagreement map", fontweight="bold")
    ax.text(0.02, 0.98, "c", transform=ax.transAxes, ha="left", va="top", fontweight="bold")

    fig.suptitle(
        "Static Security Region Characterization in Generator Power Space - case9mod",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
