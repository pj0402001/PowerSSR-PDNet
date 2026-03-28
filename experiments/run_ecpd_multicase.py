"""
Train EC-PDNet full-state surrogate on all available benchmark systems.

Cases:
  - WB2
  - WB5
  - case9mod
  - LMBM3_lf1p490
  - LMBM3_lf1p500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ecpd_multicase import (  # noqa: E402
    ECPDNet,
    build_dataset,
    evaluate_ecpd,
    make_loaders,
    sample_point_rows,
    split_indices,
    state_weight_vector,
    train_ecpd,
)


DEFAULT_CASES = ["WB2", "WB5", "case9mod", "LMBM3_lf1p490", "LMBM3_lf1p500"]

CASE_PRESETS: Dict[str, Dict[str, float]] = {
    "WB2": {
        "epochs": 120,
        "batch_size": 512,
        "lr": 1e-3,
        "lambda_state": 1.0,
        "lambda_mono": 0.1,
        "p_weight": 3.0,
    },
    "WB5": {
        "epochs": 140,
        "batch_size": 512,
        "lr": 1e-3,
        "lambda_state": 1.0,
        "lambda_mono": 0.0,
        "p_weight": 2.0,
    },
    "case9mod": {
        "epochs": 180,
        "batch_size": 256,
        "lr": 8e-4,
        "lambda_state": 2.0,
        "lambda_mono": 0.0,
        "p_weight": 4.0,
    },
    "LMBM3_lf1p490": {
        "epochs": 120,
        "batch_size": 512,
        "lr": 1e-3,
        "lambda_state": 1.0,
        "lambda_mono": 0.05,
        "p_weight": 2.0,
    },
    "LMBM3_lf1p500": {
        "epochs": 120,
        "batch_size": 512,
        "lr": 1e-3,
        "lambda_state": 1.0,
        "lambda_mono": 0.05,
        "p_weight": 2.0,
    },
}


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


def _flatten_point_rows(rows: List[Dict]) -> pd.DataFrame:
    flat = []
    for r in rows:
        base = {
            "case": r.get("case"),
            "traditional_secure": r.get("traditional_secure"),
            "model_prob_secure": r.get("model_prob_secure"),
            "threshold": r.get("threshold"),
            "model_secure": r.get("model_secure"),
            "agreement": r.get("agreement"),
            "state_mae_overall": r.get("state_mae_overall"),
        }
        inp = r.get("input") or {}
        for k, v in inp.items():
            base[f"input_{k}"] = v

        pred = r.get("state_pred") or {}
        true = r.get("state_true") or {}
        for k, v in pred.items():
            base[f"pred_{k}"] = v
        for k, v in true.items():
            base[f"true_{k}"] = v
        flat.append(base)
    return pd.DataFrame(flat)


def train_one_case(
    case_id: str,
    data_dir: Path,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    lambda_state: float,
    lambda_mono: float,
    p_weight: float,
) -> Dict:
    print("\n" + "=" * 72)
    print(f"Training EC-PDNet for {case_id}")
    print("=" * 72)

    ds = build_dataset(case_id=case_id, data_dir=data_dir, seed=seed)
    split = split_indices(ds.y_cls, seed=seed)
    loaders = make_loaders(ds, split, batch_size=batch_size)

    model = ECPDNet(dataset=ds, dropout=0.1)
    sw = state_weight_vector(ds.state_names, p_weight=p_weight)

    hist = train_ecpd(
        model=model,
        loaders=loaders,
        ds=ds,
        device=device,
        epochs=epochs,
        lr=lr,
        lambda_state=lambda_state,
        lambda_mono=lambda_mono,
        patience=30,
        state_weights=sw,
    )

    th = float(hist["best_threshold"])
    val_eval = evaluate_ecpd(model, loaders["val"], ds, device, th)
    test_eval = evaluate_ecpd(model, loaders["test"], ds, device, th)
    point_rows = sample_point_rows(
        model=model,
        ds=ds,
        split_indices=split["test"],
        device=device,
        threshold=th,
        seed=seed,
        n_each=3,
    )

    out = {
        "dataset": {
            "case": case_id,
            "n_total": int(len(ds.X_norm)),
            "n_secure": int(ds.y_cls.sum()),
            "n_insecure": int((ds.y_cls < 0.5).sum()),
            "split": {k: int(len(v)) for k, v in split.items()},
            "input_names": list(ds.input_names),
            "state_names": list(ds.state_names),
        },
        "train": {
            "epochs_run": int(len(hist["train_total"])),
            "best_threshold": th,
            "history_tail": {
                "train_total": hist["train_total"][-5:],
                "val_total": hist["val_total"][-5:],
                "val_f1@0.5": hist["val_f1@0.5"][-5:],
                "val_state_mae": hist["val_state_mae"][-5:],
                "train_mono": hist.get("train_mono", [])[-5:],
            },
        },
        "validation": val_eval,
        "test": test_eval,
        "points": point_rows,
    }

    return {"model": model, "metrics": out}


def main():
    parser = argparse.ArgumentParser(description="Run EC-PDNet on all benchmark cases")
    parser.add_argument("--data-dir", type=str, default=r"D:\安全域\1")
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-state", type=float, default=1.0)
    parser.add_argument("--lambda-mono", type=float, default=0.1)
    parser.add_argument("--p-weight", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--use-case-presets",
        action="store_true",
        help="Use tuned per-case hyperparameter presets for multicase runs.",
    )
    args = parser.parse_args()

    _set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    data_dir = Path(args.data_dir)
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    all_summary = {}
    for case_id in args.cases:
        if args.use_case_presets and case_id in CASE_PRESETS:
            cfg = CASE_PRESETS[case_id]
            epochs = int(cfg["epochs"])
            batch_size = int(cfg["batch_size"])
            lr = float(cfg["lr"])
            lambda_state = float(cfg["lambda_state"])
            lambda_mono = float(cfg["lambda_mono"])
            p_weight = float(cfg["p_weight"])
        else:
            epochs = args.epochs
            batch_size = args.batch_size
            lr = args.lr
            lambda_state = args.lambda_state
            lambda_mono = args.lambda_mono
            p_weight = args.p_weight

        print(
            f"Config[{case_id}] epochs={epochs}, batch={batch_size}, lr={lr}, "
            f"lambda_state={lambda_state}, lambda_mono={lambda_mono}, p_weight={p_weight}"
        )

        run = train_one_case(
            case_id=case_id,
            data_dir=data_dir,
            device=device,
            seed=args.seed,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            lambda_state=lambda_state,
            lambda_mono=lambda_mono,
            p_weight=p_weight,
        )
        model = run["model"]
        metrics = run["metrics"]

        ckpt = results_dir / f"{case_id}_ecpd_multicase.pth"
        metric_json = results_dir / f"{case_id}_ecpd_multicase_metrics.json"
        points_json = results_dir / f"{case_id}_ecpd_multicase_points.json"
        points_csv = results_dir / f"{case_id}_ecpd_multicase_points.csv"

        torch.save({"state_dict": model.state_dict(), "case": case_id}, ckpt)
        with open(metric_json, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable({k: v for k, v in metrics.items() if k != "points"}), f, indent=2, ensure_ascii=False)
        with open(points_json, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(metrics["points"]), f, indent=2, ensure_ascii=False)
        _flatten_point_rows(metrics["points"]).to_csv(points_csv, index=False, encoding="utf-8-sig")

        t = metrics["test"]
        print(
            f"{case_id}: F1={t['classification']['f1']:.4f}, "
            f"Acc={t['classification']['acc']:.4f}, "
            f"StateMAE={t['state']['overall_mae']:.4f}, "
            f"ClosureAbsMean={t['closure']['abs_mean']:.4f}"
        )

        all_summary[case_id] = {
            "f1": t["classification"]["f1"],
            "acc": t["classification"]["acc"],
            "state_mae": t["state"]["overall_mae"],
            "closure_abs_mean": t["closure"]["abs_mean"],
        }

    summary_path = results_dir / "ecpd_multicase_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(all_summary), f, indent=2, ensure_ascii=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
