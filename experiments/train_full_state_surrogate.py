"""
Train and evaluate a full-output surrogate for case9mod.

Outputs include:
- security feasibility probability and classification metrics
- state-variable regression metrics (P/Q/V/theta)
- pointwise comparison report against traditional IPOPT states
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from state_surrogate import (  # noqa: E402
    EnergyClosurePDNet,
    FullStatePDNet,
    build_case9mod_dataset,
    build_state_weight_vector,
    compare_points,
    evaluate_energy_consistency,
    evaluate_full_state_model,
    export_checkpoint,
    make_dataloaders,
    sample_demo_points,
    split_indices,
    train_full_state_model,
)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.int32, np.int64, np.uint32, np.uint64)):
        return int(x)
    return x


def _parse_points(raw: List[str]) -> np.ndarray:
    points: List[List[float]] = []
    for item in raw:
        chunks = [s.strip() for s in item.replace(";", " ").split() if s.strip()]
        for ch in chunks:
            if "," not in ch:
                raise ValueError(f"Invalid point format: {ch}. Use p2,p3")
            a, b = ch.split(",", 1)
            points.append([float(a), float(b)])
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train case9 full-state surrogate")
    parser.add_argument("--data-dir", type=str, default=r"D:\安全域\1", help="Traditional CSV directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-state", type=float, default=1.0)
    parser.add_argument("--lambda-voltage", type=float, default=0.05)
    parser.add_argument("--lambda-monotonic", type=float, default=0.0)
    parser.add_argument("--p1-weight", type=float, default=3.0)
    parser.add_argument(
        "--arch",
        type=str,
        default="baseline",
        choices=["baseline", "energy_closure"],
        help="Model architecture: baseline full-state head or energy-closure head",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--point",
        action="append",
        default=[],
        help="Custom query point as p2,p3. Can be repeated.",
    )
    args = parser.parse_args()

    _set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print("Loading case9mod traditional dataset...")
    dataset = build_case9mod_dataset(Path(args.data_dir), seed=args.seed)
    print(
        f"Dataset size={len(dataset.X_norm)} | "
        f"secure={int(dataset.y_cls.sum())} | insecure={int((dataset.y_cls < 0.5).sum())}"
    )

    split = split_indices(dataset.y_cls, seed=args.seed)
    loaders = make_dataloaders(dataset, split, batch_size=args.batch_size)

    if args.arch == "energy_closure":
        model = EnergyClosurePDNet(
            input_dim=2,
            n_state=len(dataset.state_names),
            x_mean=dataset.x_mean,
            x_std=dataset.x_std,
            state_mean=dataset.state_mean,
            state_std=dataset.state_std,
            total_load_mw=dataset.total_load_mw,
            dropout=0.1,
        )
    else:
        model = FullStatePDNet(input_dim=2, n_state=len(dataset.state_names), dropout=0.1)

    state_w = build_state_weight_vector(
        dataset.state_names,
        p1_weight=args.p1_weight,
        q_weight=1.2,
        v_weight=1.0,
        theta_weight=1.0,
    )

    print("Training full-state surrogate...")
    history = train_full_state_model(
        model=model,
        loaders=loaders,
        dataset=dataset,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        lambda_state=args.lambda_state,
        lambda_voltage=args.lambda_voltage,
        lambda_monotonic=args.lambda_monotonic,
        state_weight=state_w,
    )

    best_threshold = float(history["best_threshold"])
    print(f"Best threshold from validation: {best_threshold:.3f}")

    val_eval = evaluate_full_state_model(model, loaders["val"], dataset, device, best_threshold)
    test_eval = evaluate_full_state_model(model, loaders["test"], dataset, device, best_threshold)
    val_energy = evaluate_energy_consistency(model, loaders["val"], dataset, device)
    test_energy = evaluate_energy_consistency(model, loaders["test"], dataset, device)

    # Pointwise report: sampled demo + user custom points
    demo_points = sample_demo_points(dataset, seed=args.seed, n_each=3)
    custom_points = _parse_points(args.point)
    if len(custom_points) > 0:
        query_points = np.vstack([demo_points, custom_points]).astype(np.float32)
    else:
        query_points = demo_points
    point_rows = compare_points(model, query_points, dataset, device, best_threshold)

    out_metrics = {
        "dataset": {
            "n_total": int(len(dataset.X_norm)),
            "n_secure": int(dataset.y_cls.sum()),
            "n_insecure": int((dataset.y_cls < 0.5).sum()),
            "split": {k: int(len(v)) for k, v in split.items()},
            "state_names": list(dataset.state_names),
        },
        "train": {
            "epochs_run": int(len(history["train_total"])),
            "best_threshold": best_threshold,
            "arch": args.arch,
            "lambda_monotonic": float(args.lambda_monotonic),
            "p1_weight": float(args.p1_weight),
            "history_tail": {
                "train_total": history["train_total"][-5:],
                "val_total": history["val_total"][-5:],
                "val_f1@0.5": history["val_f1@0.5"][-5:],
                "val_state_mae": history["val_state_mae"][-5:],
                "train_mono": history.get("train_mono", [])[-5:],
            },
        },
        "validation": val_eval,
        "test": test_eval,
        "energy_consistency": {
            "validation": val_energy,
            "test": test_energy,
        },
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    suffix = "ecpd" if args.arch == "energy_closure" else "pdnet"
    ckpt_path = results_dir / f"case9mod_fullstate_{suffix}.pth"
    metrics_path = results_dir / f"case9mod_fullstate_{suffix}_metrics.json"
    points_path = results_dir / f"case9mod_fullstate_{suffix}_point_comparison.json"

    export_checkpoint(model, dataset, ckpt_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(out_metrics), f, indent=2, ensure_ascii=False)
    with open(points_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(point_rows), f, indent=2, ensure_ascii=False)

    print("Saved artifacts:")
    print(f"  {ckpt_path}")
    print(f"  {metrics_path}")
    print(f"  {points_path}")

    t_cls = test_eval["classification"]
    t_state = test_eval["state"]
    print("\nTest summary:")
    print(
        f"  classification: acc={t_cls['acc']:.4f} f1={t_cls['f1']:.4f} "
        f"prec={t_cls['prec']:.4f} rec={t_cls['rec']:.4f} spec={t_cls['spec']:.4f}"
    )
    print(
        f"  state MAE overall={t_state['overall_mae']:.4f} | "
        f"P={t_state['mae_group'].get('p_slack', float('nan')):.4f}, "
        f"Q={t_state['mae_group'].get('q', float('nan')):.4f}, "
        f"V={t_state['mae_group'].get('v', float('nan')):.4f}, "
        f"theta={t_state['mae_group'].get('theta', float('nan')):.4f}"
    )


if __name__ == "__main__":
    main()
