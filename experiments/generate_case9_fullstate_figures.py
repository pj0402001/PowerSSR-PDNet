"""
Generate case9 full-state boundary/probability figures from latest checkpoint.

Outputs:
  - figures/case9mod_fullstate_prob_hist.png
  - figures/case9mod_fullstate_prob_map_test.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_surrogate import (  # noqa: E402
    EnergyClosurePDNet,
    _collect_predictions,
    build_case9mod_dataset,
    make_dataloaders,
    split_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate case9 full-state figures")
    parser.add_argument("--data-dir", type=str, default=r"D:\安全域\1")
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "results" / "case9mod_fullstate_ecpd.pth"))
    parser.add_argument("--metrics", type=str, default=str(ROOT / "results" / "case9mod_fullstate_ecpd_metrics.json"))
    parser.add_argument("--bg-multiplier", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ds = build_case9mod_dataset(Path(args.data_dir), seed=args.seed, bg_multiplier=args.bg_multiplier)
    split = split_indices(ds.y_cls, seed=args.seed)
    loaders = make_dataloaders(ds, split, batch_size=args.batch_size)

    with open(args.metrics, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    th = float(metrics.get("train", {}).get("best_threshold", 0.5))

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = EnergyClosurePDNet(
        input_dim=2,
        n_state=len(ck["state_names"]),
        x_mean=np.asarray(ck["x_mean"], dtype=np.float32),
        x_std=np.asarray(ck["x_std"], dtype=np.float32),
        state_mean=np.asarray(ck["state_mean"], dtype=np.float32),
        state_std=np.asarray(ck["state_std"], dtype=np.float32),
        total_load_mw=float(ds.total_load_mw),
        dropout=0.1,
    ).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()

    pred = _collect_predictions(model, loaders["test"], device)
    probs = pred["probs"].astype(np.float64)
    labels = pred["labels"].astype(np.int64)
    boundary = pred.get("boundary", np.zeros_like(labels, dtype=np.float32)) > 0.5

    ids_test = split["test"]
    xy = ds.X_raw[ids_test]
    p2 = xy[:, 0]
    p3 = xy[:, 1]

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Figure 1: probability histogram
    pos = probs[labels == 1]
    neg = probs[labels == 0]
    neg_bd = probs[(labels == 0) & boundary]

    plt.figure(figsize=(8.8, 5.6))
    bins = np.linspace(0.0, 1.0, 61)
    plt.hist(neg, bins=bins, alpha=0.55, color="#d62728", label="Infeasible (all)")
    if neg_bd.size > 0:
        plt.hist(neg_bd, bins=bins, histtype="step", linewidth=2.0, color="#ff7f0e", label="Infeasible (boundary)")
    plt.hist(pos, bins=bins, alpha=0.55, color="#2ca02c", label="Feasible")
    plt.axvline(th, color="black", linestyle="--", linewidth=1.6, label=f"Threshold={th:.2f}")
    plt.xlabel("Predicted secure probability")
    plt.ylabel("Count")
    plt.title("case9mod full-state probability polarization (test set)")
    plt.legend(frameon=True)
    plt.tight_layout()
    out_hist = fig_dir / "case9mod_fullstate_prob_hist.png"
    plt.savefig(out_hist, dpi=300)
    plt.close()

    # Figure 2: test-space probability map (scatter)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), constrained_layout=True)

    ax = axes[0]
    ax.scatter(p2[labels == 0], p3[labels == 0], s=7, c="#d62728", alpha=0.42, label="Infeasible")
    ax.scatter(p2[labels == 1], p3[labels == 1], s=8, c="#2ca02c", alpha=0.60, label="Feasible")
    ax.set_title("Traditional labels (test split)")
    ax.set_xlabel("P_G2 (MW)")
    ax.set_ylabel("P_G3 (MW)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    sc = ax.scatter(p2, p3, c=probs, s=9, cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.85)
    if np.any((labels == 0) & boundary):
        ax.scatter(
            p2[(labels == 0) & boundary],
            p3[(labels == 0) & boundary],
            facecolors="none",
            edgecolors="white",
            linewidths=0.5,
            s=18,
            alpha=0.9,
            label="Infeasible boundary",
        )
        ax.legend(loc="best", fontsize=8)
    ax.set_title("EC-PDNet predicted secure probability")
    ax.set_xlabel("P_G2 (MW)")
    ax.set_ylabel("P_G3 (MW)")
    ax.grid(alpha=0.2)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("p_secure")

    out_map = fig_dir / "case9mod_fullstate_prob_map_test.png"
    fig.savefig(out_map, dpi=300)
    plt.close(fig)

    print(f"Saved: {out_hist}")
    print(f"Saved: {out_map}")


if __name__ == "__main__":
    main()
