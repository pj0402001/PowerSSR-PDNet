"""
Generate a case9 security-margin heatmap inspired by Eq. (28) in:
Practical Dynamic Security Region Model: A Hybrid Physical Model-Driven
and Data-Driven Approach.

Adaptation used here for case9 static security domain:
- Build the traditional secure/insecure mask from D:\\安全域\\1\\ac_opf_9results.csv.
- Compute nearest-boundary distance d(x) for each operating point x.
- Let D_ref be the nearest-boundary distance of a reference operating point.
- Define CI_d(x) = |d(x)| / D_ref (single-boundary adaptation of Eq. (28d)).
- Plot signed margin SM(x) = +CI_d(x) inside secure region, -CI_d(x) outside.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]


def _build_secure_mask(
    csv_path: Path,
    p2_axis: np.ndarray,
    p3_axis: np.ndarray,
    offset_x: int,
    offset_y: int,
) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if "p2_mw" not in df.columns or "p3_mw" not in df.columns:
        raise ValueError(f"Missing p2_mw/p3_mw in {csv_path}")

    p2_base = p2_axis[offset_x:]
    p3_base = p3_axis[offset_y:]
    if len(p2_base) < 2 or len(p3_base) < 2:
        raise RuntimeError("Axis length is too short")

    d2 = float(p2_base[1] - p2_base[0])
    d3 = float(p3_base[1] - p3_base[0])
    ix = np.rint((df["p2_mw"].to_numpy(dtype=np.float64) - float(p2_base[0])) / d2).astype(int)
    iy = np.rint((df["p3_mw"].to_numpy(dtype=np.float64) - float(p3_base[0])) / d3).astype(int)

    valid = (ix >= 0) & (ix < len(p2_base)) & (iy >= 0) & (iy < len(p3_base))
    secure = np.zeros((len(p3_axis), len(p2_axis)), dtype=bool)
    secure[iy[valid] + offset_y, ix[valid] + offset_x] = True
    return secure


def _boundary_mask(secure: np.ndarray) -> np.ndarray:
    m = secure.astype(np.int16)
    p = np.pad(m, 1, mode="constant", constant_values=0)
    n4 = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    boundary_secure = secure & (n4 < 4)
    boundary_insecure = (~secure) & (n4 > 0)
    return boundary_secure | boundary_insecure


def _boundary_points_from_contour(p2_axis: np.ndarray, p3_axis: np.ndarray, secure: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(p2_axis, p3_axis)
    fig, ax = plt.subplots(figsize=(4, 3))
    cs = ax.contour(xx, yy, secure.astype(np.float32), levels=[0.5], colors="none")
    segs = cs.allsegs[0] if len(cs.allsegs) > 0 else []
    plt.close(fig)

    verts = [seg.astype(np.float32) for seg in segs if seg is not None and len(seg) > 0]
    if verts:
        return np.vstack(verts)

    # Fallback to cell-center boundary points when contour extraction fails.
    bmask = _boundary_mask(secure)
    return np.column_stack([xx[bmask], yy[bmask]]).astype(np.float32)


def _min_distance_to_boundary(points: np.ndarray, boundary_points: np.ndarray, chunk: int = 4096) -> np.ndarray:
    if len(boundary_points) == 0:
        raise RuntimeError("Boundary is empty; cannot compute security margin")
    out = np.empty((len(points),), dtype=np.float32)
    for s in range(0, len(points), chunk):
        e = min(s + chunk, len(points))
        block = points[s:e]
        d2 = ((block[:, None, :] - boundary_points[None, :, :]) ** 2).sum(axis=2)
        out[s:e] = np.sqrt(d2.min(axis=1), dtype=np.float32)
    return out


def _pick_reference_index(
    secure: np.ndarray,
    dist_abs: np.ndarray,
    p2_axis: np.ndarray,
    p3_axis: np.ndarray,
    ref_p2: float,
    ref_p3: float,
) -> Tuple[int, int, Optional[str]]:
    ix = int(np.argmin(np.abs(p2_axis - float(ref_p2))))
    iy = int(np.argmin(np.abs(p3_axis - float(ref_p3))))
    if secure[iy, ix] and float(dist_abs[iy, ix]) > 1e-8:
        return iy, ix, None

    secure_pos = secure & (dist_abs > 1e-8)
    yy, xx = np.where(secure_pos)
    if yy.size > 0:
        d2 = (p2_axis[xx] - float(ref_p2)) ** 2 + (p3_axis[yy] - float(ref_p3)) ** 2
        k = int(np.argmin(d2))
        msg = "Requested reference is on/near boundary; moved to nearest interior secure point"
        return int(yy[k]), int(xx[k]), msg

    yy, xx = np.where(secure)
    if yy.size == 0:
        raise RuntimeError("No secure points found in traditional dataset")
    d2 = (p2_axis[xx] - float(ref_p2)) ** 2 + (p3_axis[yy] - float(ref_p3)) ** 2
    k = int(np.argmin(d2))
    msg = "All secure points are boundary points under current discretization; using nearest secure point"
    return int(yy[k]), int(xx[k]), msg


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate case9 security-margin heatmap")
    parser.add_argument("--data-dir", type=str, default=r"D:\安全域\1")
    parser.add_argument("--csv", type=str, default="ac_opf_9results.csv")
    parser.add_argument("--out-fig", type=str, default=str(ROOT / "figures" / "case9mod_security_margin_heatmap.png"))
    parser.add_argument("--out-csv", type=str, default=str(ROOT / "results" / "case9mod_security_margin_grid.csv"))
    parser.add_argument("--ref-p2", type=float, default=163.0)
    parser.add_argument("--ref-p3", type=float, default=85.0)
    parser.add_argument("--include-zero-band", dest="include_zero_band", action="store_true")
    parser.add_argument("--no-zero-band", dest="include_zero_band", action="store_false")
    parser.set_defaults(include_zero_band=True)
    parser.add_argument("--include-pred-boundary", action="store_true")
    parser.add_argument("--metrics", type=str, default=str(ROOT / "results" / "case9mod_boundaryloop_metrics.json"))
    parser.add_argument("--probs", type=str, default=str(ROOT / "results" / "case9mod_boundaryloop_probs.npy"))
    args = parser.parse_args()

    p2_base = np.linspace(10.0, 300.0, 300, dtype=np.float32)
    p3_base = np.linspace(10.0, 270.0, 300, dtype=np.float32)

    if bool(args.include_zero_band):
        p2_axis = np.concatenate([np.array([0.0], dtype=np.float32), p2_base])
        p3_axis = np.concatenate([np.array([0.0], dtype=np.float32), p3_base])
        offset_x, offset_y = 1, 1
    else:
        p2_axis = p2_base
        p3_axis = p3_base
        offset_x, offset_y = 0, 0

    csv_path = Path(args.data_dir) / args.csv
    secure = _build_secure_mask(csv_path, p2_axis, p3_axis, offset_x=offset_x, offset_y=offset_y)

    XX, YY = np.meshgrid(p2_axis, p3_axis)
    pts = np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32)
    bd_pts = _boundary_points_from_contour(p2_axis, p3_axis, secure)

    dist = _min_distance_to_boundary(pts, bd_pts).reshape(len(p3_axis), len(p2_axis))
    signed = np.where(secure, dist, -dist).astype(np.float32)

    ref_iy, ref_ix, ref_msg = _pick_reference_index(
        secure,
        dist,
        p2_axis,
        p3_axis,
        ref_p2=float(args.ref_p2),
        ref_p3=float(args.ref_p3),
    )
    d_ref = float(abs(signed[ref_iy, ref_ix]))
    if d_ref < 1e-8:
        pos_secure = dist[secure & (dist > 1e-8)]
        if pos_secure.size > 0:
            d_ref = float(np.percentile(pos_secure, 50.0))
            ref_msg = "Reference margin was near zero; normalized by median interior secure distance"
        else:
            d_ref = float(min(float(p2_base[1] - p2_base[0]), float(p3_base[1] - p3_base[0])))
            ref_msg = "No interior secure point found; normalized by one-grid-step distance"

    ci_abs = (np.abs(signed) / d_ref).astype(np.float32)
    sm_signed = np.where(secure, ci_abs, -ci_abs).astype(np.float32)

    vmax = float(max(1.0, np.percentile(ci_abs, 99.0)))
    levels = np.linspace(-vmax, vmax, 61)

    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    cf = ax.contourf(XX, YY, sm_signed, levels=levels, cmap="RdYlGn", extend="both")

    ax.contour(XX, YY, secure.astype(np.float32), levels=[0.5], colors="black", linewidths=1.6)

    legend_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.6, label="Traditional boundary"),
    ]

    if bool(args.include_pred_boundary):
        probs_path = Path(args.probs)
        metrics_path = Path(args.metrics)
        if probs_path.exists() and metrics_path.exists():
            probs_base = np.load(probs_path).astype(np.float32)
            with open(metrics_path, "r", encoding="utf-8") as f:
                th = float(json.load(f).get("best_threshold", 0.5))
            if probs_base.size == 300 * 300:
                pred = np.zeros_like(secure, dtype=np.float32)
                pred[offset_y:, offset_x:] = (probs_base.reshape(300, 300) > th).astype(np.float32)
                cs = ax.contour(XX, YY, pred, levels=[0.5], colors="white", linewidths=1.5, linestyles="--")
                for coll in cs.collections:
                    coll.set_path_effects([pe.Stroke(linewidth=2.6, foreground="#222222"), pe.Normal()])
                legend_handles.insert(
                    0,
                    Line2D([0], [0], color="white", linestyle="--", linewidth=1.5, label="Predicted boundary"),
                )

    ax.scatter(
        [float(p2_axis[ref_ix])],
        [float(p3_axis[ref_iy])],
        marker="*",
        s=130,
        c="#2b83ba",
        edgecolors="white",
        linewidths=0.9,
        zorder=8,
        label="Reference point",
    )
    legend_handles.append(
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#2b83ba", markeredgecolor="white", markersize=10, linestyle="", label="Reference point")
    )

    ax.set_xlim(float(p2_axis[0]), float(p2_axis[-1]))
    ax.set_ylim(float(p3_axis[0]), float(p3_axis[-1]))
    ax.set_xlabel("P_G2 (MW)")
    ax.set_ylabel("P_G3 (MW)")
    ax.set_title("Case9 security-margin heatmap (Eq.28-inspired)")
    ax.grid(alpha=0.2)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    cb = fig.colorbar(cf, ax=ax)
    cb.set_label("Signed security margin (CI_d normalized by D_ref)")

    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(
        {
            "p2_mw": XX.ravel(),
            "p3_mw": YY.ravel(),
            "is_secure": secure.astype(np.int8).ravel(),
            "security_margin_signed": sm_signed.ravel(),
            "security_margin_abs": ci_abs.ravel(),
        }
    )
    out_df.to_csv(out_csv, index=False)

    print(f"Saved figure: {out_fig}")
    print(f"Saved grid data: {out_csv}")
    print(f"Reference point: ({float(p2_axis[ref_ix]):.3f}, {float(p3_axis[ref_iy]):.3f})")
    print(f"D_ref (MW): {d_ref:.6f}")
    if ref_msg is not None:
        print(f"Note: {ref_msg}")


if __name__ == "__main__":
    main()
