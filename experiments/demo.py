"""
Quick smoke test and demo visualization for the SSR-PDNet framework.
Runs a fast experiment on IEEE 9-bus for verifying the pipeline works.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

from power_system import generate_security_region_data, get_test_network, get_network_info
from models import BaselineNN, PhysicsNN, SSR_PDNet
from trainer import make_data_loaders, train_baseline, train_ssr_pdnet, evaluate_model
from visualization import (plot_security_region_2d, plot_training_curves,
                           plot_roc_pr_curves, plot_comparison_bar, plot_confusion_matrix)

FIG_DIR = Path('../figures')
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR = Path('../data')
DATA_DIR.mkdir(exist_ok=True)
SAVE_DIR = Path('../results')
SAVE_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


def run_demo(case: str = 'case9', n_samples: int = 2000, epochs: int = 100):
    print(f"\n{'='*60}")
    print(f"  DEMO: {case.upper()} | {n_samples} samples | {epochs} epochs")
    print(f"{'='*60}")

    # ─── Data generation ──────────────────────────
    print("\n[1] Generating dataset (Latin Hypercube)...")
    X, y, meta = generate_security_region_data(
        case=case, n_samples=n_samples, load_variation=0.45,
        method='latin_hypercube', random_seed=42
    )
    print(f"  X: {X.shape}, feasibility rate: {y.mean():.3f}")
    np.save(DATA_DIR / f'{case}_X.npy', X)
    np.save(DATA_DIR / f'{case}_y.npy', y)

    # 2D grid for visualization
    print("\n[2] Generating 2D grid for visualization...")
    n_grid = 40
    X_g, y_g, meta_g = generate_security_region_data(
        case=case, n_samples=n_grid**2, load_variation=0.45,
        method='grid', random_seed=42
    )
    np.save(DATA_DIR / f'{case}_X_grid.npy', X_g)
    np.save(DATA_DIR / f'{case}_y_grid.npy', y_g)
    np.save(DATA_DIR / f'{case}_P_grid.npy', meta_g['P_raw'])
    print(f"  Grid: {X_g.shape}, feasibility rate: {y_g.mean():.3f}")

    # ─── Data loaders ─────────────────────────────
    train_loader, val_loader, test_loader = make_data_loaders(
        X, y, val_ratio=0.15, test_ratio=0.15, batch_size=256, balance=True
    )

    input_dim = X.shape[1]
    net = get_test_network(case)
    n_bus = get_network_info(net)['n_bus']

    # ─── Train models ─────────────────────────────
    print("\n[3] Training models...")
    histories = {}
    test_results = {}

    # A) Baseline
    print("  Training Baseline NN...")
    baseline = BaselineNN(input_dim=input_dim, hidden_dims=[128, 128, 64], dropout=0.1)
    histories['Baseline'] = train_baseline(
        baseline, train_loader, val_loader, epochs=epochs, lr=1e-3, patience=25, device=DEVICE
    )
    test_results['Baseline'] = evaluate_model(baseline, test_loader, DEVICE)
    print(f"  Baseline: acc={test_results['Baseline']['acc']:.4f}, f1={test_results['Baseline']['f1']:.4f}")

    # B) Physics-NN
    print("  Training Physics-NN...")
    phys_nn = PhysicsNN(input_dim=input_dim, hidden_dims=[256, 256, 128], n_bus=n_bus)
    histories['Physics-NN'] = train_baseline(
        phys_nn, train_loader, val_loader, epochs=epochs, lr=1e-3, patience=25, device=DEVICE
    )
    test_results['Physics-NN'] = evaluate_model(phys_nn, test_loader, DEVICE)
    print(f"  Physics-NN: acc={test_results['Physics-NN']['acc']:.4f}, f1={test_results['Physics-NN']['f1']:.4f}")

    # C) SSR-PDNet (proposed)
    print("  Training SSR-PDNet (proposed)...")
    ssr = SSR_PDNet(
        input_dim=input_dim, feature_dim=128,
        classifier_dims=[256, 256, 128], physics_dims=[128, 64],
        n_bus=n_bus, dropout=0.1, use_physics_head=True
    )
    histories['SSR-PDNet'] = train_ssr_pdnet(
        ssr, train_loader, val_loader,
        epochs=int(epochs * 1.2), lr=1e-3, lr_dual=1e-2,
        lambda_physics=0.1, lambda_contrastive=0.05,
        patience=30, device=DEVICE,
    )
    test_results['SSR-PDNet'] = evaluate_model(ssr, test_loader, DEVICE)
    print(f"  SSR-PDNet: acc={test_results['SSR-PDNet']['acc']:.4f}, f1={test_results['SSR-PDNet']['f1']:.4f}")

    # ─── Visualizations ───────────────────────────
    print("\n[4] Generating visualizations...")

    # Training curves
    plot_training_curves(histories, save_path=str(FIG_DIR / f'{case}_training.png'))
    print(f"  Saved: {FIG_DIR}/{case}_training.png")

    # ROC / PR
    plot_roc_pr_curves(test_results, save_path=str(FIG_DIR / f'{case}_roc_pr.png'))
    print(f"  Saved: {FIG_DIR}/{case}_roc_pr.png")

    # Bar comparison
    plot_comparison_bar(test_results, save_path=str(FIG_DIR / f'{case}_comparison.png'))
    print(f"  Saved: {FIG_DIR}/{case}_comparison.png")

    # Confusion matrices
    plot_confusion_matrix(test_results, save_path=str(FIG_DIR / f'{case}_confusion.png'))
    print(f"  Saved: {FIG_DIR}/{case}_confusion.png")

    # 2D Security Region (SSR-PDNet)
    X_g_t = torch.FloatTensor(X_g).to(DEVICE)
    ssr.eval()
    with torch.no_grad():
        logits_g, _ = ssr(X_g_t)
        probs_g = torch.sigmoid(logits_g).cpu().numpy()

    P_grid_raw = np.load(DATA_DIR / f'{case}_P_grid.npy')
    fig = plot_security_region_2d(
        P_grid=P_grid_raw,
        Q_grid=P_grid_raw,
        labels_true=y_g,
        probs_pred=probs_g,
        case_name=case,
        load_idx=(0, 1),
        p_base=meta['p_base'],
        save_path=str(FIG_DIR / f'{case}_security_region.png')
    )
    print(f"  Saved: {FIG_DIR}/{case}_security_region.png")

    # ─── Print final summary ──────────────────────
    print("\n" + "="*50)
    print(f"RESULTS SUMMARY — {case.upper()}")
    print("="*50)
    print(f"{'Model':15s} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Spec':>8}")
    print("-"*55)
    for name, res in test_results.items():
        print(f"{name:15s} {res['acc']:8.4f} {res['f1']:8.4f} "
              f"{res['prec']:8.4f} {res['rec']:8.4f} {res['spec']:8.4f}")

    # Save model weights
    torch.save(ssr.state_dict(), SAVE_DIR / f'{case}_ssr_pdnet.pth')
    print(f"\nSSR-PDNet model saved: {SAVE_DIR}/{case}_ssr_pdnet.pth")

    return test_results, histories


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='case9', choices=['case9', 'case30', 'case57', 'case118'])
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    run_demo(case=args.case, n_samples=args.samples, epochs=args.epochs)
