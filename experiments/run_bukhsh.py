"""
Main experiment script for Bukhsh et al. (2013) test cases.
Uses WB2 (analytical), WB5, LMBM3, case9mod with pandapower.
Runs all three models: Baseline, Physics-NN, SSR-DL.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from bukhsh_cases import get_bukhsh_case
from bukhsh_data import generate_bukhsh_data, generate_bukhsh_grid
from models import BaselineNN, PhysicsNN, SSR_DL
from trainer import make_data_loaders, train_baseline, train_ssr_dl, evaluate_model
from visualization import (
    plot_security_region_2d, plot_training_curves, plot_roc_pr_curves,
    plot_comparison_bar, plot_confusion_matrix,
)

# ─── Directories ────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / 'figures'
DATA_DIR = ROOT / 'data'
SAVE_DIR = ROOT / 'results'
for d in [FIG_DIR, DATA_DIR, SAVE_DIR]:
    d.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ─── Experiment configuration ───────────────────────────────
CONFIGS = {
    # WB2: 2-bus analytical; tiny feasibility region (~2.5%)
    # Load space: [P2, Q2] ∈ [175–525 MW, -175 to -525 MVAR]
    'WB2': {
        'n_samples': 4000, 'n_grid': 60,
        'load_variation': 0.5,          # ±50% of P2_nom=350 MW
        'epochs': 150, 'batch': 256,
        'hidden': [128, 128, 64],
        'feature_dim': 64,
        'classifier_dims': [128, 128, 64],
        'physics_dims': [64, 32],
    },
    # WB5: 5-bus meshed; feasibility boundary hit ~40x for two loads
    # Use Load2 vs Load3 as the primary 2D visualization plane
    'WB5': {
        'n_samples': 5000, 'n_grid': 45,
        'load_variation': 19.0,         # gives ~0.05x to 20x range → ~50% feasibility
        'epochs': 200, 'batch': 512,
        'hidden': [256, 256, 128, 64],
        'feature_dim': 128,
        'classifier_dims': [256, 256, 128],
        'physics_dims': [128, 64],
        'grid_load_idx': (1, 2),        # Load2 vs Load3
    },
    # LMBM3: skip (infeasible at all sampled points due to structural line overload)
    # case9mod: modified IEEE 9-bus; Qmin=-5 MVAR (tight); loads at 60% base
    # 0.1x–5.0x gives ~50% feasibility rate
    'case9mod': {
        'n_samples': 8000, 'n_grid': 50,
        'load_variation': 4.9,          # 0.1x – 5.0x (center at 2.5x base)
        'epochs': 250, 'batch': 512,
        'hidden': [512, 512, 256, 128],
        'feature_dim': 256,
        'classifier_dims': [512, 512, 256, 128],
        'physics_dims': [256, 128],
    },
}


def run_case(case_name: str, cfg: dict, quick: bool = False):
    print(f"\n{'='*65}")
    print(f"  Case: {case_name}")
    print(f"{'='*65}")

    epochs = max(50, cfg['epochs'] // 3) if quick else cfg['epochs']
    n_samples = max(1500, cfg['n_samples'] // 3) if quick else cfg['n_samples']
    n_grid = 35 if quick else cfg['n_grid']

    # ─── 1. Data generation ───────────────────────────────
    print(f"\n[1] Data generation ({n_samples} samples)...")
    X, y, meta = generate_bukhsh_data(
        case_name, n_samples=n_samples,
        load_variation=cfg['load_variation'], seed=42,
    )
    np.save(DATA_DIR / f'{case_name}_X.npy', X)
    np.save(DATA_DIR / f'{case_name}_y.npy', y)
    meta_save = {k: (v.tolist() if hasattr(v, 'tolist') else v)
                 for k, v in meta.items()
                 if k not in ['P_raw', 'Q_raw', 'violations']}
    with open(DATA_DIR / f'{case_name}_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta_save, f, indent=2, ensure_ascii=False)
    print(f"  Feasibility rate: {y.mean():.3f}")

    # 2D grid for visualization (use 2 load dimensions)
    n_load_half = meta['n_load']  # n_load is number of loads (P-features)
    load_idx = cfg.get('grid_load_idx', (0, 1) if n_load_half >= 2 else (0, 0))

    print(f"\n[2] 2D grid ({n_grid}x{n_grid}) for visualization...")
    X_g, y_g, meta_g = generate_bukhsh_grid(
        case_name, n_per_dim=n_grid,
        load_variation=cfg['load_variation'],
        load_idx=load_idx if case_name != 'WB2' else (0, 1),
    )
    np.save(DATA_DIR / f'{case_name}_X_grid.npy', X_g)
    np.save(DATA_DIR / f'{case_name}_y_grid.npy', y_g)
    if 'P_grid' in meta_g:
        np.save(DATA_DIR / f'{case_name}_P_grid.npy', meta_g['P_grid'])
        np.save(DATA_DIR / f'{case_name}_Q_grid.npy', meta_g['Q_grid'])
    print(f"  Grid feasibility rate: {y_g.mean():.3f}")

    # ─── 2. Data loaders ──────────────────────────────────
    train_loader, val_loader, test_loader = make_data_loaders(
        X, y, val_ratio=0.15, test_ratio=0.15,
        batch_size=cfg['batch'], balance=True, seed=42,
    )
    input_dim = X.shape[1]
    n_bus = meta['n_bus']

    # ─── 3. Model training ────────────────────────────────
    print(f"\n[3] Training models (epochs={epochs})...")
    histories, test_results = {}, {}

    # A) Baseline NN
    print("  [A] Baseline NN...")
    baseline = BaselineNN(input_dim=input_dim, hidden_dims=cfg['hidden'], dropout=0.1)
    histories['Baseline'] = train_baseline(
        baseline, train_loader, val_loader,
        epochs=epochs, lr=1e-3, patience=30, device=DEVICE,
    )
    test_results['Baseline'] = evaluate_model(baseline, test_loader, DEVICE)
    r = test_results['Baseline']
    print(f"  Baseline: acc={r['acc']:.4f} f1={r['f1']:.4f} prec={r['prec']:.4f} rec={r['rec']:.4f}")

    # B) Physics-informed NN
    print("  [B] Physics-NN...")
    phys = PhysicsNN(input_dim=input_dim, hidden_dims=cfg['hidden'], n_bus=n_bus, dropout=0.1)
    histories['Physics-NN'] = train_baseline(
        phys, train_loader, val_loader,
        epochs=epochs, lr=1e-3, patience=30, device=DEVICE,
    )
    test_results['Physics-NN'] = evaluate_model(phys, test_loader, DEVICE)
    r = test_results['Physics-NN']
    print(f"  Physics-NN: acc={r['acc']:.4f} f1={r['f1']:.4f}")

    # C) SSR-DL (proposed)
    print("  [C] SSR-DL (proposed)...")
    ssr = SSR_DL(
        input_dim=input_dim,
        feature_dim=cfg['feature_dim'],
        classifier_dims=cfg['classifier_dims'],
        physics_dims=cfg['physics_dims'],
        n_bus=n_bus, dropout=0.1, use_physics_head=True,
    )
    histories['SSR-DL'] = train_ssr_dl(
        ssr, train_loader, val_loader,
        epochs=int(epochs * 1.2), lr=1e-3, lr_dual=1e-2,
        lambda_physics=0.1, lambda_contrastive=0.05,
        patience=35, device=DEVICE,
    )
    test_results['SSR-DL'] = evaluate_model(ssr, test_loader, DEVICE)
    r = test_results['SSR-DL']
    print(f"  SSR-DL: acc={r['acc']:.4f} f1={r['f1']:.4f}")

    # ─── 4. Save models ───────────────────────────────────
    for mname, model in [('baseline', baseline), ('physics_nn', phys), ('ssr_dl', ssr)]:
        torch.save(model.state_dict(), SAVE_DIR / f'{case_name}_{mname}.pth')

    # Save metrics
    metrics_save = {name: {k: v for k, v in res.items() if k not in ['probs', 'labels']}
                    for name, res in test_results.items()}
    with open(SAVE_DIR / f'{case_name}_metrics.json', 'w') as f:
        json.dump(metrics_save, f, indent=2)

    # ─── 5. Visualizations ────────────────────────────────
    print(f"\n[4] Generating figures...")

    plot_training_curves(histories, save_path=str(FIG_DIR / f'{case_name}_training.png'))
    plot_roc_pr_curves(test_results, save_path=str(FIG_DIR / f'{case_name}_roc_pr.png'))
    plot_comparison_bar(test_results, save_path=str(FIG_DIR / f'{case_name}_comparison.png'))
    plot_confusion_matrix(test_results, save_path=str(FIG_DIR / f'{case_name}_confusion.png'))

    # 2D security region (SSR-DL predictions on grid)
    X_g_t = torch.FloatTensor(X_g).to(DEVICE)
    ssr.eval()
    with torch.no_grad():
        logits_g, _ = ssr(X_g_t)
        probs_g = torch.sigmoid(logits_g).cpu().numpy()

    # Also get Baseline predictions
    baseline.eval()
    with torch.no_grad():
        bl_logits = baseline(X_g_t)
        probs_bl = torch.sigmoid(bl_logits).cpu().numpy()

    # Use SSR-DL for the main 2D security region figure
    if case_name == 'WB2':
        # WB2: 2D, directly use P_raw and Q_raw columns
        P_plot = meta_g['P_raw']   # 2D meshgrid
        Q_plot = meta_g['Q_raw']
    else:
        P_grid_raw = np.load(DATA_DIR / f'{case_name}_P_grid.npy')
        Q_grid_raw = np.load(DATA_DIR / f'{case_name}_Q_grid.npy')
        P_plot = P_grid_raw
        Q_plot = Q_grid_raw

    _plot_2d_ssr(
        case_name=case_name,
        P_grid=P_plot, Q_grid=Q_plot,
        y_grid=y_g, probs_ssr=probs_g, probs_bl=probs_bl,
        n_per_dim=n_grid, meta=meta, meta_g=meta_g,
        load_idx=load_idx if case_name != 'WB2' else (0, 1),
    )

    print(f"\n  All figures saved to {FIG_DIR}/")
    return test_results, histories


def _plot_2d_ssr(case_name, P_grid, Q_grid, y_grid, probs_ssr, probs_bl,
                 n_per_dim, meta, meta_g, load_idx):
    """Plot comprehensive 2D security region figure for a case."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({
        'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
        'figure.dpi': 150, 'savefig.dpi': 300,
        'axes.grid': True, 'grid.alpha': 0.25,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    green_red = LinearSegmentedColormap.from_list('gr', ['#d62728', '#2ca02c'])
    FEAS = '#2ca02c'; INFEAS = '#d62728'

    labels_2d = y_grid.reshape(n_per_dim, n_per_dim)
    probs_ssr_2d = probs_ssr.reshape(n_per_dim, n_per_dim)
    probs_bl_2d = probs_bl.reshape(n_per_dim, n_per_dim)

    if case_name == 'WB2':
        P1_arr = meta_g['P_arr']; P2_arr = meta_g['Q_arr']
        xlabel = 'Load Active Power P₂ (MW)'
        ylabel = 'Load Reactive Power Q₂ (MVAR)'
    else:
        P1_arr = meta_g['P1_arr']; P2_arr = meta_g['P2_arr']
        i0, i1 = meta_g.get('load_idx', (0, 1))
        xlabel = f'Load {i0+1} Active Power P (MW)'
        ylabel = f'Load {i1+1} Active Power P (MW)'

    PG, QG = np.meshgrid(P1_arr, P2_arr)

    def plot_panel(ax, data_2d, title, colormap=None, is_prob=False):
        if is_prob:
            cs = ax.contourf(PG, QG, data_2d, levels=50, cmap=colormap, alpha=0.85, vmin=0, vmax=1)
            plt.colorbar(cs, ax=ax, label='P(Secure)')
            ax.contour(PG, QG, data_2d, levels=[0.5], colors='black', linewidths=2.0)
        else:
            ax.contourf(PG, QG, data_2d, levels=[-0.5, 0.5, 1.5],
                        colors=[INFEAS, FEAS], alpha=0.7)
            ax.contour(PG, QG, data_2d, levels=[0.5], colors='black', linewidths=2.0)

        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=8)

        # Base operating point marker
        if case_name == 'WB2':
            pb = [350.0, -350.0]
        else:
            pb = meta['p_base']
        if pb is not None and len(pb) >= 2:
            px = pb[0] if case_name == 'WB2' else pb[load_idx[0]]
            py = pb[1] if case_name == 'WB2' else pb[load_idx[1]]
            ax.scatter([px], [py], marker='*', s=300, color='gold', zorder=10,
                       edgecolors='black', linewidths=0.8, label='Nominal point')
            ax.legend(fontsize=9, loc='upper left')

    # Panel 1: Ground Truth
    plot_panel(axes[0], labels_2d, f'Ground Truth SSR\n({case_name})')
    patches = [mpatches.Patch(color=FEAS, label='Secure (Feasible)'),
               mpatches.Patch(color=INFEAS, label='Insecure (Infeasible)')]
    axes[0].legend(handles=patches, fontsize=9, loc='lower right')

    # Panel 2: SSR-DL prediction
    plot_panel(axes[1], probs_ssr_2d, 'SSR-DL Predicted\nSecurity Region',
               colormap=green_red, is_prob=True)
    axes[1].contour(PG, QG, labels_2d, levels=[0.5], colors='white',
                    linewidths=1.5, linestyles='--')
    axes[1].plot([], [], 'k-', lw=2, label='Predicted boundary')
    axes[1].plot([], [], 'w--', lw=1.5, label='True boundary')
    axes[1].legend(fontsize=9, loc='lower right')

    # Panel 3: Error map
    preds_ssr = (probs_ssr_2d > 0.5).astype(float)
    errors = np.abs(preds_ssr - labels_2d)
    acc_test = 1 - errors.mean()
    axes[2].contourf(PG, QG, labels_2d, levels=[-0.5, 0.5, 1.5],
                     colors=[INFEAS, FEAS], alpha=0.25)
    if errors.any():
        axes[2].contourf(PG, QG, errors, levels=[0.5, 1.5],
                         colors=['darkred'], alpha=0.65)
    axes[2].contour(PG, QG, labels_2d, levels=[0.5], colors='black', linewidths=1.5)
    axes[2].contour(PG, QG, probs_ssr_2d, levels=[0.5], colors='#ff7f0e', linewidths=1.5)
    axes[2].set_xlabel(xlabel, fontweight='bold')
    axes[2].set_ylabel(ylabel, fontweight='bold')
    axes[2].set_title(f'SSR-DL Error Map\n(Grid Acc = {acc_test:.3f})', fontweight='bold', pad=8)
    err_patch = mpatches.Patch(color='darkred', alpha=0.7, label='Misclassified region')
    axes[2].legend(handles=[err_patch], fontsize=9)
    if case_name != 'WB2':
        pb = meta['p_base']
        px = pb[load_idx[0]]; py = pb[load_idx[1]]
        axes[2].scatter([px], [py], marker='*', s=300, color='gold', zorder=10,
                        edgecolors='black', linewidths=0.8)

    fig.suptitle(
        f'Static Security Region (SSR) Characterization — {case_name}\n'
        f'[Bukhsh et al. 2013 Test Case]',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    fig.savefig(str(FIG_DIR / f'{case_name}_security_region.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved: {FIG_DIR}/{case_name}_security_region.png")


def print_summary(all_results: dict):
    print("\n" + "=" * 75)
    print("FINAL RESULTS SUMMARY — Bukhsh et al. (2013) Test Cases")
    print("=" * 75)
    header = f"{'Case':12s} {'Model':14s} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Spec':>8}"
    print(header)
    print("-" * 75)
    for case_name, results in all_results.items():
        for i, (model_name, res) in enumerate(results.items()):
            prefix = case_name if i == 0 else " " * len(case_name)
            print(f"{prefix:12s} {model_name:14s} "
                  f"{res['acc']:8.4f} {res['f1']:8.4f} "
                  f"{res['prec']:8.4f} {res['rec']:8.4f} {res['spec']:8.4f}")
        print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SSR-DL experiments on Bukhsh cases')
    parser.add_argument('--cases', nargs='+', default=['WB2', 'WB5', 'LMBM3', 'case9mod'],
                        choices=list(CONFIGS.keys()))
    parser.add_argument('--quick', action='store_true', help='Quick run with fewer samples/epochs')
    args = parser.parse_args()

    all_results = {}
    for case in args.cases:
        t0 = time.time()
        results, histories = run_case(case, CONFIGS[case], quick=args.quick)
        all_results[case] = results
        print(f"\n  Wall time for {case}: {time.time()-t0:.1f}s")

    print_summary(all_results)
