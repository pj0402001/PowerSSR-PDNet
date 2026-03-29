"""
Benchmark runtime comparison for case9mod.

Compares:
1) Traditional AC-OPF pointwise solve time recorded in ac_opf_9results.csv
2) EC-PDNet inference latency on the same operating points

Outputs:
- results/case9mod_runtime_benchmark.json
- results/case9mod_runtime_benchmark.csv
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from state_surrogate import EnergyClosurePDNet  # noqa: E402


def _stats_seconds(values: np.ndarray) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {
            "n": 0,
            "mean_s": float("nan"),
            "median_s": float("nan"),
            "p95_s": float("nan"),
            "min_s": float("nan"),
            "max_s": float("nan"),
            "sum_s": float("nan"),
        }
    return {
        "n": int(v.size),
        "mean_s": float(v.mean()),
        "median_s": float(np.median(v)),
        "p95_s": float(np.percentile(v, 95.0)),
        "min_s": float(v.min()),
        "max_s": float(v.max()),
        "sum_s": float(v.sum()),
    }


def _load_model(
    ckpt_path: Path,
    total_load_mw: float,
    device: torch.device,
) -> Tuple[EnergyClosurePDNet, np.ndarray, np.ndarray]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    state_mean = np.asarray(ckpt["state_mean"], dtype=np.float32)
    state_std = np.asarray(ckpt["state_std"], dtype=np.float32)
    state_names = list(ckpt["state_names"])

    model = EnergyClosurePDNet(
        input_dim=2,
        n_state=len(state_names),
        x_mean=x_mean,
        x_std=x_std,
        state_mean=state_mean,
        state_std=state_std,
        total_load_mw=float(total_load_mw),
        dropout=0.1,
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, x_mean, x_std


def _inference_total_seconds(
    model: EnergyClosurePDNet,
    x_norm: torch.Tensor,
    batch_size: int,
) -> float:
    n = int(x_norm.shape[0])
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = x_norm[i : i + batch_size]
            logits, state = model(xb)
            _ = torch.sigmoid(logits)
            _ = state
    if x_norm.device.type == "cuda":
        torch.cuda.synchronize()
    return float(time.perf_counter() - start)


def _benchmark(
    model: EnergyClosurePDNet,
    x_norm: torch.Tensor,
    batch_size: int,
    repeats: int,
    warmup: int,
) -> Dict[str, float]:
    for _ in range(max(warmup, 0)):
        _inference_total_seconds(model, x_norm, batch_size=batch_size)

    times = []
    for _ in range(max(repeats, 1)):
        times.append(_inference_total_seconds(model, x_norm, batch_size=batch_size))

    arr = np.asarray(times, dtype=np.float64)
    n = int(x_norm.shape[0])
    mean_s = float(arr.mean())
    std_s = float(arr.std())

    return {
        "n_points": n,
        "batch_size": int(batch_size),
        "repeats": int(repeats),
        "warmup": int(warmup),
        "total_mean_s": mean_s,
        "total_std_s": std_s,
        "total_min_s": float(arr.min()),
        "total_max_s": float(arr.max()),
        "per_point_mean_ms": float(mean_s * 1000.0 / max(n, 1)),
        "throughput_points_per_s": float(n / max(mean_s, 1e-12)),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark case9 traditional vs EC-PDNet runtime")
    parser.add_argument("--data-dir", type=str, default=r"D:\安全域\1")
    parser.add_argument(
        "--csv",
        type=str,
        default=r"D:\安全域\1\ac_opf_9results.csv",
        help="Traditional case9 CSV containing calculation_time",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(ROOT / "results" / "case9mod_fullstate_ecpd.pth"),
        help="EC-PDNet checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for throughput test")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=0, help="Set torch CPU threads when > 0")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = ["p2_mw", "p3_mw", "calculation_time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    points = df[["p2_mw", "p3_mw"]].to_numpy(dtype=np.float32)
    trad_times = df["calculation_time"].dropna().to_numpy(dtype=np.float64)
    trad_stats = _stats_seconds(trad_times)

    total_load_mw = float(df["total_load"].iloc[0]) if "total_load" in df.columns else 189.0
    model, x_mean, x_std = _load_model(Path(args.ckpt), total_load_mw=total_load_mw, device=device)

    x_norm_np = (points - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)
    x_norm = torch.from_numpy(x_norm_np.astype(np.float32)).to(device)

    dl_batch1 = _benchmark(
        model=model,
        x_norm=x_norm,
        batch_size=1,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    dl_batchn = _benchmark(
        model=model,
        x_norm=x_norm,
        batch_size=max(1, args.batch_size),
        repeats=args.repeats,
        warmup=args.warmup,
    )

    trad_per_point_ms = float(trad_stats["mean_s"] * 1000.0)
    speedup_single = trad_per_point_ms / max(dl_batch1["per_point_mean_ms"], 1e-12)
    speedup_batch = trad_per_point_ms / max(dl_batchn["per_point_mean_ms"], 1e-12)

    result = {
        "meta": {
            "case": "case9mod",
            "n_points": int(len(points)),
            "device": str(device),
            "torch_version": str(torch.__version__),
            "python_version": str(platform.python_version()),
            "platform": str(platform.platform()),
            "cpu": str(platform.processor()),
            "threads": int(torch.get_num_threads()),
            "ckpt": str(Path(args.ckpt)),
            "traditional_csv": str(csv_path),
            "notes": [
                "Traditional timing uses calculation_time from CSV (feasible points in traditional scan output).",
                "DL timing excludes model loading and data I/O; only forward inference is measured.",
            ],
        },
        "traditional": trad_stats,
        "ecpd_inference": {
            "single_point": dl_batch1,
            "batched": dl_batchn,
        },
        "speedup": {
            "traditional_vs_single_point": float(speedup_single),
            "traditional_vs_batched": float(speedup_batch),
        },
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_json = results_dir / "case9mod_runtime_benchmark.json"
    out_csv = results_dir / "case9mod_runtime_benchmark.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    rows = [
        {
            "method": "Traditional IPOPT (from CSV)",
            "n_points": int(trad_stats["n"]),
            "mean_per_point_ms": float(trad_stats["mean_s"] * 1000.0),
            "p95_per_point_ms": float(trad_stats["p95_s"] * 1000.0),
            "throughput_points_per_s": float(1.0 / max(trad_stats["mean_s"], 1e-12)),
            "total_time_s": float(trad_stats["sum_s"]),
            "speedup_vs_traditional": 1.0,
        },
        {
            "method": "EC-PDNet inference (batch=1)",
            "n_points": int(dl_batch1["n_points"]),
            "mean_per_point_ms": float(dl_batch1["per_point_mean_ms"]),
            "p95_per_point_ms": float("nan"),
            "throughput_points_per_s": float(dl_batch1["throughput_points_per_s"]),
            "total_time_s": float(dl_batch1["total_mean_s"]),
            "speedup_vs_traditional": float(speedup_single),
        },
        {
            "method": f"EC-PDNet inference (batch={max(1, args.batch_size)})",
            "n_points": int(dl_batchn["n_points"]),
            "mean_per_point_ms": float(dl_batchn["per_point_mean_ms"]),
            "p95_per_point_ms": float("nan"),
            "throughput_points_per_s": float(dl_batchn["throughput_points_per_s"]),
            "total_time_s": float(dl_batchn["total_mean_s"]),
            "speedup_vs_traditional": float(speedup_batch),
        },
    ]
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("Case9 runtime benchmark complete.")
    print(f"Traditional mean per-point: {trad_per_point_ms:.4f} ms")
    print(f"EC-PDNet batch=1 per-point: {dl_batch1['per_point_mean_ms']:.4f} ms | speedup: {speedup_single:.1f}x")
    print(
        f"EC-PDNet batch={max(1, args.batch_size)} per-point: {dl_batchn['per_point_mean_ms']:.4f} ms | "
        f"speedup: {speedup_batch:.1f}x"
    )
    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
