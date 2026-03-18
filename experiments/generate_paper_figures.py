"""
Generate comprehensive summary visualization for the paper.
Creates multi-panel figures comparing all models across all Bukhsh cases.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, auc
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / 'figures'
DATA_DIR = ROOT / 'data'
SAVE_DIR = ROOT / 'results'
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Serif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 11,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'axes.grid': True, 'grid.alpha': 0.25,
    'axes.spines.top': False, 'axes.spines.right': False,
})

FEAS_COLOR = '#2ca02c'
INFEAS_COLOR = '#d62728'
MODEL_COLORS = {'Baseline': '#1f77b4', 'Physics-NN': '#ff7f0e', 'SSR-DL': '#2ca02c'}
MODEL_LS = {'Baseline': '-', 'Physics-NN': '--', 'SSR-DL': '-.'}


def load_case_results(case_name):
    """Load saved results for a case."""
    results_file = SAVE_DIR / f'{case_name}_metrics.json'
    if not results_file.exists():
        return None
    with open(results_file) as f:
        return json.load(f)


def make_summary_table():
    """Print LaTeX-ready results table."""
    cases = ['WB2', 'case9mod']
    models = ['Baseline', 'Physics-NN', 'SSR-DL']
    metrics = ['acc', 'f1', 'prec', 'rec', 'spec']

    print("\n% LaTeX results table:")
    print("\\begin{tabular}{llccccc}")
    print("\\hline")
    print("Case & Model & Accuracy & F1 Score & Precision & Recall & Specificity \\\\")
    print("\\hline")
    for case in cases:
        results = load_case_results(case)
        if results is None:
            continue
        for i, model in enumerate(models):
            if model not in results:
                continue
            r = results[model]
            case_label = case if i == 0 else ""
            row = f"{case_label} & {model}"
            for m in metrics:
                row += f" & {r.get(m, 0):.4f}"
            row += " \\\\"
            print(row)
        print("\\hline")
    print("\\end{tabular}")


def plot_wb2_analytical():
    """
    Special WB2 visualization: show the two local solutions
    and the feasibility region in (P, Q) load space.
    """
    sys.path.insert(0, str(ROOT / 'src'))
    from bukhsh_data import generate_WB2_grid, _wb2_power_flow

    # Generate fine grid
    print("Generating WB2 fine grid for analytical visualization...")
    n = 80
    X_g, y_g, meta_g = generate_WB2_grid(n_per_dim=n, load_variation=0.5)

    P_arr = meta_g['P_arr']  # MW
    Q_arr = meta_g['Q_arr']  # MVAR (negative = capacitive)
    V2_2d = meta_g['V2'].reshape(n, n)
    labels_2d = y_g.reshape(n, n)
    PG, QG = np.meshgrid(P_arr, Q_arr)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Feasibility region with V2 contours
    ax = axes[0]
    ax.contourf(PG, QG, labels_2d, levels=[-0.5, 0.5, 1.5],
                colors=[INFEAS_COLOR, FEAS_COLOR], alpha=0.65)
    ax.contour(PG, QG, labels_2d, levels=[0.5], colors='black', linewidths=2.5)
    # Voltage contours
    v_levels = [0.95, 0.97, 0.99, 1.01, 1.03, 1.05]
    cs = ax.contour(PG, QG, V2_2d, levels=v_levels, colors='gray',
                    linewidths=0.8, linestyles=':')
    ax.clabel(cs, fmt='%.2f p.u.', fontsize=7)
    ax.scatter([350], [-350], marker='*', s=300, color='gold',
               zorder=10, edgecolors='black', lw=0.8, label='Nominal op. point')
    patches = [mpatches.Patch(color=FEAS_COLOR, label='Feasible (SSR)'),
               mpatches.Patch(color=INFEAS_COLOR, label='Infeasible')]
    ax.legend(handles=patches + ax.get_legend_handles_labels()[0], fontsize=8, loc='upper left')
    ax.set_xlabel('Load Active Power P₂ (MW)', fontweight='bold')
    ax.set_ylabel('Load Reactive Power Q₂ (MVAR)', fontweight='bold')
    ax.set_title('WB2: True SSR with V₂ Contours\n(Voltage constraint: 0.95–1.05 p.u.)',
                 fontweight='bold')

    # Panel 2: Two local solutions in (P, Q) space
    # For each point, find which solution branch the NR converges to
    V2_low = np.full((n, n), np.nan)
    V2_high = np.full((n, n), np.nan)
    baseMVA = 100.0
    from scipy.optimize import fsolve
    z = 0.04 + 0.20j
    g, b_val = (1/z).real, (1/z).imag
    V1 = 0.964

    def pf_eq(vars, P2, Q2):
        V2, th2 = vars
        Pcalc = V2**2 * g - V2*V1*(g*np.cos(th2) + b_val*np.sin(th2))
        Qcalc = -V2**2 * b_val - V2*V1*(g*np.sin(th2) - b_val*np.cos(th2))
        return [Pcalc - P2, Qcalc - Q2]

    print("Finding dual solutions for WB2...")
    for j, P2 in enumerate(P_arr):
        for i, Q2 in enumerate(Q_arr):
            P_inj = -P2 / baseMVA
            Q_inj = -Q2 / baseMVA
            sols = []
            for v0, th0 in [(0.7, -1.5), (1.1, -0.4), (0.9, -0.8), (1.2, -0.3)]:
                try:
                    sol, _, ier, _ = fsolve(pf_eq, [v0, th0], args=(P_inj, Q_inj), full_output=True)
                    if ier == 1 and sol[0] > 0:
                        resid = abs(np.array(pf_eq(sol, P_inj, Q_inj))).max()
                        if resid < 1e-7:
                            is_new = all(abs(s[0]-sol[0]) > 0.01 or abs(s[1]-sol[1]) > 0.01 for s in sols)
                            if is_new:
                                sols.append(sol)
                except:
                    pass
            if len(sols) >= 2:
                vs = sorted([s[0] for s in sols])
                V2_low[i, j] = vs[0]
                V2_high[i, j] = vs[1]
            elif len(sols) == 1:
                V2_low[i, j] = sols[0][0]

    ax = axes[1]
    # Show where two solutions exist
    dual_mask = ~np.isnan(V2_high)
    ax.contourf(PG, QG, labels_2d, levels=[-0.5, 0.5, 1.5],
                colors=[INFEAS_COLOR, FEAS_COLOR], alpha=0.3)
    if dual_mask.any():
        # Color by V2_high - V2_low (distance between solutions)
        gap = V2_high - V2_low
        gap[~dual_mask] = np.nan
        cs2 = ax.contourf(PG, QG, V2_high, levels=20, cmap='Blues', alpha=0.7)
        plt.colorbar(cs2, ax=ax, label='High-V solution V₂ (p.u.)')
        ax.contour(PG, QG, labels_2d, levels=[0.5], colors='red', linewidths=2.0)
        # Mark dual-solution region
        ax.contour(PG, QG, dual_mask.astype(float), levels=[0.5],
                   colors='purple', linewidths=1.5, linestyles='--')
        ax.plot([], [], color='purple', linestyle='--', lw=1.5, label='Dual-solution boundary')
    ax.scatter([350], [-350], marker='*', s=300, color='gold', zorder=10, edgecolors='black', lw=0.8)
    ax.set_xlabel('Load Active Power P₂ (MW)', fontweight='bold')
    ax.set_ylabel('Load Reactive Power Q₂ (MVAR)', fontweight='bold')
    ax.set_title('WB2: High-Voltage Solution Branch\n(Multiple local optima region)',
                 fontweight='bold')
    ax.legend(fontsize=8)

    # Panel 3: SSR-DL prediction vs ground truth
    ax = axes[2]
    # Load model
    ssr_model_path = SAVE_DIR / 'WB2_ssr_dl.pth'
    if not ssr_model_path.exists():
        # Try alternate naming
        ssr_model_path = list(SAVE_DIR.glob('WB2*ssr*pth'))
        if ssr_model_path:
            ssr_model_path = ssr_model_path[0]
        else:
            ax.text(0.5, 0.5, 'Model not found', ha='center', va='center')
            plt.tight_layout()
            fig.savefig(str(FIG_DIR / 'WB2_analytical.png'), bbox_inches='tight')
            plt.close(fig)
            return

    from models import SSR_DL
    ssr = SSR_DL(input_dim=2, feature_dim=64, classifier_dims=[128, 128, 64],
                 physics_dims=[64, 32], n_bus=2)
    try:
        ssr.load_state_dict(torch.load(str(ssr_model_path), map_location='cpu', weights_only=True))
    except:
        ssr.load_state_dict(torch.load(str(ssr_model_path), map_location='cpu'))
    ssr.eval()

    X_g_t = torch.FloatTensor(X_g)
    with torch.no_grad():
        logits, _ = ssr(X_g_t)
        probs_g = torch.sigmoid(logits).numpy()

    probs_2d = probs_g.reshape(n, n)
    green_red = LinearSegmentedColormap.from_list('gr', [INFEAS_COLOR, FEAS_COLOR])
    cs = ax.contourf(PG, QG, probs_2d, levels=50, cmap=green_red, alpha=0.85, vmin=0, vmax=1)
    plt.colorbar(cs, ax=ax, label='P(Secure | Load)')
    ax.contour(PG, QG, probs_2d, levels=[0.5], colors='black', linewidths=2.0)
    ax.contour(PG, QG, labels_2d, levels=[0.5], colors='white', linewidths=1.5, linestyles='--')
    ax.scatter([350], [-350], marker='*', s=300, color='gold', zorder=10, edgecolors='black', lw=0.8)
    ax.plot([], [], 'k-', lw=2, label='SSR-DL boundary')
    ax.plot([], [], 'w--', lw=1.5, label='True boundary')
    ax.legend(fontsize=8)

    preds = (probs_2d > 0.5).astype(float)
    acc = 1 - np.abs(preds - labels_2d).mean()
    ax.set_xlabel('Load Active Power P₂ (MW)', fontweight='bold')
    ax.set_ylabel('Load Reactive Power Q₂ (MVAR)', fontweight='bold')
    ax.set_title(f'SSR-DL Prediction on WB2\n(Grid accuracy = {acc:.3f})', fontweight='bold')

    fig.suptitle('WB2 (Bukhsh 2013): 2-Bus System with Multiple Local OPF Solutions\n'
                 'Static Security Region Characterization via Deep Learning',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(FIG_DIR / 'WB2_analytical.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {FIG_DIR}/WB2_analytical.png")


def plot_final_comparison():
    """Comprehensive model comparison across all cases."""
    cases_with_data = []
    for case in ['WB2', 'WB5', 'case9mod']:
        r = load_case_results(case)
        if r is not None:
            cases_with_data.append((case, r))

    if not cases_with_data:
        print("No results found.")
        return

    n_cases = len(cases_with_data)
    fig = plt.figure(figsize=(16, 5 * n_cases))
    gs = GridSpec(n_cases, 4, figure=fig, hspace=0.45, wspace=0.35)

    metric_labels = ['Accuracy', 'F1', 'Precision', 'Recall', 'Specificity']
    metric_keys = ['acc', 'f1', 'prec', 'rec', 'spec']
    model_names = ['Baseline', 'Physics-NN', 'SSR-DL']

    for row, (case_name, results) in enumerate(cases_with_data):
        # Panel 1: Bar chart of metrics
        ax = fig.add_subplot(gs[row, 0])
        x = np.arange(len(metric_labels))
        n_models = len(model_names)
        width = 0.22
        for i, model in enumerate(model_names):
            if model not in results:
                continue
            vals = [results[model].get(m, 0) for m in metric_keys]
            offset = (i - n_models/2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width * 0.9,
                         label=model, color=list(MODEL_COLORS.values())[i],
                         alpha=0.85, edgecolor='white')
            for bar, val in zip(bars, vals):
                if val > 0.95:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=6, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels(['Acc', 'F1', 'Prec', 'Rec', 'Spec'], fontsize=9)
        ax.set_ylim([0.85, 1.05])
        ax.set_ylabel('Score')
        ax.set_title(f'{case_name}: Performance Metrics', fontweight='bold')
        if row == 0:
            ax.legend(fontsize=8, loc='lower right')

        # Panel 2: Confusion matrix style heatmap (2x2 per model)
        ax2 = fig.add_subplot(gs[row, 1])
        mat_data = np.zeros((len(model_names), 4))  # TP, TN, FP, FN
        for i, model in enumerate(model_names):
            if model in results:
                r = results[model]
                mat_data[i] = [r.get('tp', 0), r.get('tn', 0), r.get('fp', 0), r.get('fn', 0)]

        total = mat_data.sum(axis=1, keepdims=True) + 1e-10
        mat_frac = mat_data / total

        im = ax2.imshow(mat_frac[:, :2], cmap='Blues', vmin=0, vmax=1, aspect='auto')
        ax2.set_xticks([0, 1]); ax2.set_xticklabels(['TP rate', 'TN rate'])
        ax2.set_yticks(range(len(model_names))); ax2.set_yticklabels(model_names, fontsize=9)
        for i in range(len(model_names)):
            for j in range(2):
                ax2.text(j, i, f'{mat_frac[i,j]:.3f}', ha='center', va='center',
                        fontsize=9, fontweight='bold',
                        color='white' if mat_frac[i,j] > 0.7 else 'black')
        ax2.set_title(f'{case_name}: TP/TN Rates', fontweight='bold')

        # Panel 3: Security region visualization (if grid data available)
        ax3 = fig.add_subplot(gs[row, 2])
        X_g_path = DATA_DIR / f'{case_name}_X_grid.npy'
        y_g_path = DATA_DIR / f'{case_name}_y_grid.npy'
        if X_g_path.exists() and y_g_path.exists():
            X_g = np.load(str(X_g_path))
            y_g = np.load(str(y_g_path))
            n_sq = int(np.sqrt(len(y_g)))

            # Try to predict with SSR-DL
            from models import SSR_DL
            n_bus_map = {'WB2': 2, 'WB5': 5, 'LMBM3': 3, 'case9mod': 9}
            n_bus = n_bus_map.get(case_name, 9)
            input_dim = X_g.shape[1]
            n_feat_dim = {'WB2': 64, 'WB5': 128, 'case9mod': 256}.get(case_name, 128)
            cls_dims = {'WB2': [128,128,64], 'WB5': [256,256,128], 'case9mod': [512,512,256,128]}.get(case_name, [256,256,128])
            phys_dims = {'WB2': [64,32], 'WB5': [128,64], 'case9mod': [256,128]}.get(case_name, [128,64])

            ssr = SSR_DL(input_dim=input_dim, feature_dim=n_feat_dim,
                        classifier_dims=cls_dims, physics_dims=phys_dims, n_bus=n_bus)
            model_files = list(SAVE_DIR.glob(f'{case_name}*ssr*pth'))
            if model_files:
                try:
                    ssr.load_state_dict(torch.load(str(model_files[0]), map_location='cpu', weights_only=True))
                except:
                    ssr.load_state_dict(torch.load(str(model_files[0]), map_location='cpu'))
                ssr.eval()
                with torch.no_grad():
                    logits, _ = ssr(torch.FloatTensor(X_g))
                    probs = torch.sigmoid(logits).numpy()

                labels_2d = y_g.reshape(n_sq, n_sq)
                probs_2d = probs.reshape(n_sq, n_sq)

                green_red = LinearSegmentedColormap.from_list('gr', [INFEAS_COLOR, FEAS_COLOR])
                cs = ax3.contourf(probs_2d, levels=30, cmap=green_red, alpha=0.85, vmin=0, vmax=1)
                ax3.contour(probs_2d, levels=[0.5], colors='black', linewidths=2.0)
                ax3.contour(labels_2d, levels=[0.5], colors='white', linewidths=1.5, linestyles='--')
                ax3.plot([], [], 'k-', lw=2, label='SSR-DL boundary')
                ax3.plot([], [], 'w--', lw=1.5, label='True boundary')
                acc_g = 1 - np.abs((probs_2d > 0.5).astype(float) - labels_2d).mean()
                ax3.set_title(f'{case_name}: SSR-DL Prediction\n(Grid acc={acc_g:.3f})', fontweight='bold')
                ax3.legend(fontsize=8)
                ax3.set_xlabel('Load Dim 1')
                ax3.set_ylabel('Load Dim 2')
            else:
                ax3.text(0.5, 0.5, 'Model not found', ha='center', va='center')

        # Panel 4: Training F1 curves
        ax4 = fig.add_subplot(gs[row, 3])
        for hist_file in ROOT.glob(f'figures/{case_name}_training.png'):
            pass  # Will use text fallback
        # Try to load from numpy
        ax4.text(0.5, 0.5, f'{case_name}\nResults loaded\nfrom JSON',
                ha='center', va='center', fontsize=9)
        ax4.set_title(f'{case_name}: Summary', fontweight='bold')

        # Show best results
        best_f1 = max(results.get(m, {}).get('f1', 0) for m in model_names if m in results)
        best_acc = max(results.get(m, {}).get('acc', 0) for m in model_names if m in results)
        ssr_f1 = results.get('SSR-DL', {}).get('f1', 0)
        text = f"Best F1: {best_f1:.4f}\nBest Acc: {best_acc:.4f}\nSSR-DL F1: {ssr_f1:.4f}"
        ax4.text(0.5, 0.4, text, ha='center', va='center', fontsize=10,
                transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    fig.suptitle('Static Security Region Characterization — Bukhsh et al. (2013) Test Cases\n'
                 'Comparison: Baseline NN vs Physics-NN vs SSR-DL (Proposed)',
                 fontsize=13, fontweight='bold', y=0.98)

    fig.savefig(str(FIG_DIR / 'final_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {FIG_DIR}/final_comparison.png")


if __name__ == '__main__':
    print("Generating WB2 analytical visualization...")
    plot_wb2_analytical()
    print("\nGenerating final comparison figure...")
    plot_final_comparison()
    print("\nGenerating LaTeX table...")
    make_summary_table()
    print("\nDone!")
