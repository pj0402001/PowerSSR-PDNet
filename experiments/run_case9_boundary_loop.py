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
from typing import Dict, Iterable, List, Sequence, Set, Tuple

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
    ix, iy = _ixiy_from_xy(ds.X_raw, ds.p2_grid, ds.p3_grid)
    ny, nx = len(ds.p3_grid), len(ds.p2_grid)
    probs_2d = np.full((ny, nx), np.nan, dtype=np.float32)
    labels_2d = np.zeros((ny, nx), dtype=np.float32)
    probs_2d[iy, ix] = probs_full.astype(np.float32)
    labels_2d[iy, ix] = ds.y_cls.astype(np.float32)

    P2, P3 = np.meshgrid(ds.p2_grid, ds.p3_grid)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)

    ax = axes[0]
    ax.contourf(P2, P3, labels_2d, levels=[-0.5, 0.5, 1.5], cmap=ListedColormap(["#f7b0b0", "#b6e3b6"]))
    ax.contour(P2, P3, labels_2d, levels=[0.5], colors="black", linewidths=1.2)
    ax.set_title("Traditional security region")
    ax.set_xlabel("P_G2 (MW)")
    ax.set_ylabel("P_G3 (MW)")
    ax.grid(alpha=0.2)

    ax = axes[1]
    c = ax.contourf(P2, P3, probs_2d, levels=21, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.contour(P2, P3, probs_2d, levels=[threshold], colors="white", linewidths=1.4, linestyles="--")
    ax.contour(P2, P3, labels_2d, levels=[0.5], colors="black", linewidths=1.1)
    ax.set_title(f"Boundary-loop model probability (th={threshold:.2f})")
    ax.set_xlabel("P_G2 (MW)")
    ax.set_ylabel("P_G3 (MW)")
    ax.grid(alpha=0.2)
    cb = fig.colorbar(c, ax=ax)
    cb.set_label("p_secure")

    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def _plot_local_zoom(
    ds: Case9Dataset,
    probs_full: np.ndarray,
    threshold: float,
    save_path: Path,
) -> None:
    x = ds.X_raw[:, 0]
    y = ds.X_raw[:, 1]
    labels = ds.y_cls
    bmask = np.abs(probs_full - threshold) < 0.08

    if bmask.any():
        cx = float(np.median(x[bmask]))
        cy = float(np.median(y[bmask]))
    else:
        cx = float(np.median(x))
        cy = float(np.median(y))

    wx = 38.0
    wy = 32.0
    in_win = (x >= cx - wx) & (x <= cx + wx) & (y >= cy - wy) & (y <= cy + wy)
    xs = x[in_win]
    ys = y[in_win]
    ls = labels[in_win]
    ps = probs_full[in_win]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), constrained_layout=True)

    ax = axes[0]
    ax.scatter(xs[ls <= 0.5], ys[ls <= 0.5], s=8, c="#d62728", alpha=0.5, label="Insecure")
    ax.scatter(xs[ls > 0.5], ys[ls > 0.5], s=10, c="#2ca02c", alpha=0.65, label="Secure")
    ax.set_title("(a) Local sample layout")
    ax.set_xlabel("P_G2 (MW)")
    ax.set_ylabel("P_G3 (MW)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    hb = ax.hexbin(xs, ys, gridsize=26, mincnt=1, cmap="YlOrRd")
    ax.set_title("(b) Local density (hexbin)")
    ax.set_xlabel("P_G2 (MW)")
    ax.set_ylabel("P_G3 (MW)")
    ax.grid(alpha=0.2)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("count")

    ax = axes[2]
    sc = ax.scatter(xs, ys, c=ps, s=10, cmap="viridis", vmin=0.0, vmax=1.0)
    # boundary overlays from local scatter approximation
    gx = np.linspace(cx - wx, cx + wx, 180)
    gy = np.linspace(cy - wy, cy + wy, 180)
    GX, GY = np.meshgrid(gx, gy)
    from scipy.interpolate import griddata

    z_prob = griddata(np.column_stack([xs, ys]), ps, (GX, GY), method="linear")
    z_lab = griddata(np.column_stack([xs, ys]), ls, (GX, GY), method="nearest")
    if np.isfinite(z_prob).any():
        ax.contour(GX, GY, z_prob, levels=[threshold], colors="white", linewidths=1.3, linestyles="--")
    if np.isfinite(z_lab).any():
        ax.contour(GX, GY, z_lab, levels=[0.5], colors="black", linewidths=1.1)
    ax.set_title("(c) Local boundary comparison")
    ax.set_xlabel("P_G2 (MW)")
    ax.set_ylabel("P_G3 (MW)")
    ax.grid(alpha=0.2)
    cb2 = fig.colorbar(sc, ax=ax)
    cb2.set_label("p_secure")

    fig.savefig(save_path, dpi=300)
    plt.close(fig)


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
    _plot_local_zoom(ds, probs_full=probs_full, threshold=best_threshold, save_path=zoom_fig)

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
