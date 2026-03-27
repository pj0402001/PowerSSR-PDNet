"""
Main experiment script for power system static security region characterization.
Runs the complete pipeline:
1. Data generation for multiple IEEE test cases
2. Model training (Baseline, PhysicsNN, SSR-PDNet)
3. Evaluation and comparison
4. Visualization of security regions
"""

import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path

from power_system import generate_security_region_data, get_test_network, get_network_info
from models import BaselineNN, PhysicsNN, SSR_PDNet
from trainer import (make_data_loaders, train_baseline, train_ssr_pdnet,
                     evaluate_model)
from visualization import (plot_security_region_2d, plot_training_curves,
                           plot_roc_pr_curves, plot_comparison_bar,
                           plot_confusion_matrix, plot_feasibility_rate_vs_load)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CASES = ['case9', 'case30', 'case118']
N_SAMPLES = {
    'case9': 4000,
    'case30': 6000,
    'case118': 8000,
}
LOAD_VARIATION = 0.5      # ±50% load variation
BATCH_SIZE = 512
EPOCHS_BASELINE = 200
EPOCHS_SSR = 250
PATIENCE = 40
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = Path('results')
FIG_DIR = Path('figures')
DATA_DIR = Path('data')

for d in [SAVE_DIR, FIG_DIR, DATA_DIR]:
    d.mkdir(exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Cases: {CASES}")
print("=" * 60)


def run_case(case: str):
    """Run full experiment for a single IEEE test case."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {case.upper()}")
    print(f"{'='*60}")

    # ─── 1. Data Generation ────────────────────────────────
    print("\n[1/4] Generating dataset...")
    t0 = time.time()

    # Main training/evaluation dataset (Latin Hypercube)
    X, y, meta = generate_security_region_data(
        case=case,
        n_samples=N_SAMPLES[case],
        load_variation=LOAD_VARIATION,
        method='latin_hypercube',
        random_seed=42,
    )
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Feasibility rate: {y.mean():.3f}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # 2D grid dataset for visualization (only for case9 and case30)
    if case in ['case9', 'case30']:
        print("  Generating 2D grid for visualization...")
        X_grid, y_grid, meta_grid = generate_security_region_data(
            case=case,
            n_samples=2500,  # 50x50 grid
            load_variation=LOAD_VARIATION,
            method='grid',
            random_seed=42,
        )
        np.save(DATA_DIR / f'{case}_X_grid.npy', X_grid)
        np.save(DATA_DIR / f'{case}_y_grid.npy', y_grid)
        np.save(DATA_DIR / f'{case}_P_grid.npy', meta_grid['P_raw'])
        print(f"  Grid: {X_grid.shape}, feasibility rate: {y_grid.mean():.3f}")

    # Save dataset
    np.save(DATA_DIR / f'{case}_X.npy', X)
    np.save(DATA_DIR / f'{case}_y.npy', y)
    with open(DATA_DIR / f'{case}_meta.json', 'w') as f:
        meta_save = {k: v.tolist() if hasattr(v, 'tolist') else v
                     for k, v in meta.items()
                     if k not in ['P_raw', 'Q_raw', 'violations']}
        json.dump(meta_save, f, indent=2)

    # ─── 2. Data Loaders ──────────────────────────────────
    train_loader, val_loader, test_loader = make_data_loaders(
        X, y, val_ratio=0.15, test_ratio=0.15, batch_size=BATCH_SIZE, balance=True
    )

    input_dim = X.shape[1]
    net = get_test_network(case)
    net_info = get_network_info(net)
    n_bus = net_info['n_bus']

    # ─── 3. Model Training ────────────────────────────────
    print("\n[2/4] Training models...")
    histories = {}
    test_results = {}

    # Model 1: Baseline NN
    print("\n  [A] Training Baseline NN...")
    baseline = BaselineNN(input_dim=input_dim, hidden_dims=[256, 256, 128, 64], dropout=0.1)
    print(f"  Baseline parameters: {sum(p.numel() for p in baseline.parameters()):,}")
    t0 = time.time()
    histories['Baseline'] = train_baseline(
        baseline, train_loader, val_loader,
        epochs=EPOCHS_BASELINE, lr=1e-3, patience=PATIENCE, device=DEVICE
    )
    print(f"  Training time: {time.time()-t0:.1f}s")

    # Model 2: Physics-Informed NN
    print("\n  [B] Training Physics-Informed NN...")
    physics_nn = PhysicsNN(input_dim=input_dim, hidden_dims=[512, 512, 256, 128], n_bus=n_bus)
    print(f"  Physics-NN parameters: {sum(p.numel() for p in physics_nn.parameters()):,}")
    t0 = time.time()
    histories['Physics-NN'] = train_baseline(
        physics_nn, train_loader, val_loader,
        epochs=EPOCHS_BASELINE, lr=1e-3, patience=PATIENCE, device=DEVICE
    )
    print(f"  Training time: {time.time()-t0:.1f}s")

    # Model 3: SSR-PDNet (proposed)
    print("\n  [C] Training SSR-PDNet (proposed)...")
    ssr_model = SSR_PDNet(
        input_dim=input_dim,
        feature_dim=256,
        classifier_dims=[512, 512, 256, 128],
        physics_dims=[256, 128],
        n_bus=n_bus,
        dropout=0.1,
        use_physics_head=True,
    )
    print(f"  SSR-PDNet parameters: {sum(p.numel() for p in ssr_model.parameters()):,}")
    t0 = time.time()
    histories['SSR-PDNet'] = train_ssr_pdnet(
        ssr_model, train_loader, val_loader,
        epochs=EPOCHS_SSR,
        lr=1e-3,
        lr_dual=1e-2,
        lambda_physics=0.1,
        lambda_boundary=0.05,
        lambda_contrastive=0.05,
        patience=PATIENCE,
        device=DEVICE,
    )
    print(f"  Training time: {time.time()-t0:.1f}s")

    # ─── 4. Evaluation ────────────────────────────────────
    print("\n[3/4] Evaluating models...")
    models = {
        'Baseline': baseline,
        'Physics-NN': physics_nn,
        'SSR-PDNet': ssr_model,
    }

    for name, model in models.items():
        result = evaluate_model(model, test_loader, DEVICE)
        test_results[name] = result
        print(f"  {name}: acc={result['acc']:.4f} f1={result['f1']:.4f} "
              f"prec={result['prec']:.4f} rec={result['rec']:.4f} "
              f"spec={result['spec']:.4f}")

    # Save results
    results_save = {name: {k: v for k, v in res.items()
                           if k not in ['probs', 'labels']}
                   for name, res in test_results.items()}
    with open(SAVE_DIR / f'{case}_results.json', 'w') as f:
        json.dump(results_save, f, indent=2)

    # Save models
    for name, model in models.items():
        torch.save(model.state_dict(),
                   SAVE_DIR / f'{case}_{name.lower().replace("-", "_")}_weights.pth')

    # ─── 5. Visualization ─────────────────────────────────
    print("\n[4/4] Generating visualizations...")

    # Training curves
    plot_training_curves(
        histories,
        save_path=str(FIG_DIR / f'{case}_training_curves.png')
    )

    # ROC and PR curves
    plot_roc_pr_curves(
        test_results,
        save_path=str(FIG_DIR / f'{case}_roc_pr.png')
    )

    # Comparison bar chart
    plot_comparison_bar(
        test_results,
        save_path=str(FIG_DIR / f'{case}_comparison.png')
    )

    # Confusion matrices
    plot_confusion_matrix(
        test_results,
        save_path=str(FIG_DIR / f'{case}_confusion.png')
    )

    # 2D Security region visualization
    if case in ['case9', 'case30']:
        X_grid = np.load(DATA_DIR / f'{case}_X_grid.npy')
        y_grid = np.load(DATA_DIR / f'{case}_y_grid.npy')
        P_grid_raw = np.load(DATA_DIR / f'{case}_P_grid.npy')

        X_grid_t = torch.FloatTensor(X_grid).to(DEVICE)
        ssr_model.eval()
        with torch.no_grad():
            logits_grid, _ = ssr_model(X_grid_t)
            probs_grid = torch.sigmoid(logits_grid).cpu().numpy()

        n_load = meta['n_load']
        plot_security_region_2d(
            P_grid=P_grid_raw,
            Q_grid=meta['Q_raw'][:len(y_grid)] if 'Q_raw' in meta else P_grid_raw,
            labels_true=y_grid,
            probs_pred=probs_grid,
            case_name=case,
            load_idx=(0, 1),
            p_base=meta['p_base'],
            save_path=str(FIG_DIR / f'{case}_security_region_2d.png')
        )

    print(f"\nCase {case} complete. Results saved to {SAVE_DIR}/")
    return histories, test_results


def main():
    all_results = {}
    feasibility_rates = []

    for case in CASES:
        try:
            hist, results = run_case(case)
            all_results[case] = results

            # Load feasibility rate from saved meta
            import json
            with open(DATA_DIR / f'{case}_meta.json') as f:
                meta = json.load(f)
            feasibility_rates.append(meta['feasibility_rate'])
        except Exception as e:
            print(f"ERROR in {case}: {e}")
            import traceback
            traceback.print_exc()

    # Cross-case summary figure
    if feasibility_rates:
        plot_feasibility_rate_vs_load(
            case_names=[c.upper() for c in CASES[:len(feasibility_rates)]],
            feasibility_rates=feasibility_rates,
            save_path=str(FIG_DIR / 'feasibility_rates.png')
        )

    # Summary table
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    for case, results in all_results.items():
        print(f"\n{case.upper()}:")
        for model_name, res in results.items():
            print(f"  {model_name:15s}: "
                  f"acc={res['acc']:.4f} f1={res['f1']:.4f} "
                  f"prec={res['prec']:.4f} rec={res['rec']:.4f}")

    print("\nAll experiments complete!")
    print(f"Figures saved to: {FIG_DIR}/")
    print(f"Results saved to: {SAVE_DIR}/")


if __name__ == '__main__':
    main()
