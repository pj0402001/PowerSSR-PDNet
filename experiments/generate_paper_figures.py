"""
Generate comprehensive summary visualization for the paper.
Creates multi-panel figures comparing all models across key Bukhsh cases.
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
MODEL_COLORS = {'Baseline': '#1f77b4', 'Physics-NN': '#ff7f0e', 'SSR-PDNet': '#2ca02c'}
MODEL_LS = {'Baseline': '-', 'Physics-NN': '--', 'SSR-PDNet': '-.'}


def load_case_results(case_name):
    """Load saved results for a case."""
    results_file = SAVE_DIR / f'{case_name}_metrics.json'
    if not results_file.exists():
        return None
    with open(results_file) as f:
        return json.load(f)


def make_summary_table():
    """Print LaTeX-ready results table."""
    cases = ['WB2', 'WB5', 'case9mod', 'LMBM3_lf1p490', 'LMBM3_lf1p500']
    models = ['Baseline', 'Physics-NN', 'SSR-PDNet']
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


def _axis_info(case_name):
    axis = {
        'WB2': ('P2 load (MW)', 'Q2 load (MVAR)', np.array([350.0, -350.0]), np.array([250.0, 350.0])),
        'WB5': ('P_G1 (MW)', 'P_G5 (MW)', np.array([350.0, 200.0]), np.array([350.0, 200.0])),
        'case9mod': ('P_G2 (MW)', 'P_G3 (MW)', np.array([163.0, 85.0]), np.array([145.0, 130.0])),
    }
    return axis.get(case_name, ('Dim 1', 'Dim 2', np.array([0.0, 0.0]), np.array([1.0, 1.0])))


def _infer_grid_shape(X_g: np.ndarray):
    """Infer (n_y, n_x) from 2D coordinate grid; return None if irregular."""
    if X_g.ndim != 2 or X_g.shape[1] < 2:
        return None
    x_unique = np.unique(np.round(X_g[:, 0].astype(np.float64), 6))
    y_unique = np.unique(np.round(X_g[:, 1].astype(np.float64), 6))
    n_x = int(len(x_unique))
    n_y = int(len(y_unique))
    if n_x > 1 and n_y > 1 and n_x * n_y == len(X_g):
        return n_y, n_x
    return None


def _to_raw_axes(case_name: str, X_g: np.ndarray) -> np.ndarray:
    """Convert normalized grids back to physical axes when needed."""
    _, _, x_mean, x_std = _axis_info(case_name)
    # WB2 grids are normalized; WB5/case9mod grids are already raw MW values.
    # Use a range-based heuristic to avoid double-scaling raw coordinates.
    q95_abs = float(np.percentile(np.abs(X_g), 95))
    if q95_abs <= 4.0:
        return X_g * x_std + x_mean
    return X_g


def _predict_probs_batched(model: torch.nn.Module, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """Run model inference in batches to avoid high peak memory."""
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.FloatTensor(X[i:i + batch_size])
            logits, _ = model(xb)
            preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds, axis=0)


def plot_wb2_analytical():
    """Generate the updated WB2 analytical figure used in the paper."""
    sys.path.insert(0, str(ROOT / 'src'))
    from bukhsh_data import generate_WB2_grid

    print("Generating WB2 fine grid for analytical visualization...")
    _, _, meta_g = generate_WB2_grid(n_per_dim=90)
    P_grid = meta_g['P_grid']
    Q_grid = meta_g['Q_grid']
    labels = meta_g['labels_2d']
    V2 = meta_g['V2']
    PG1 = meta_g['PG1']
    dual_gap = meta_g['dual_gap']
    dual_mask = meta_g['dual_mask']

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 9.0))
    fig.patch.set_facecolor('white')
    panel_specs = [
        ('a', labels, 'Security strip in load space', None),
        ('b', V2, r'Feasible-state voltage $V_2$ (p.u.)', 'viridis'),
        ('c', PG1, r'Slack generation $P_{G1}$ (MW)', 'cividis'),
        ('d', dual_gap, r'Dual-solution voltage gap $\Delta V_2$ (p.u.)', 'plasma'),
    ]

    for ax, (tag, data, title, cmap) in zip(axes.ravel(), panel_specs):
        ax.set_facecolor('#fbfaf7')
        ax.grid(True, alpha=0.18, linestyle='--', linewidth=0.6)
        if cmap is None:
            ax.contourf(P_grid, Q_grid, data, levels=[-0.5, 0.5, 1.5],
                        colors=['#efe7da', '#5b8c5a'], alpha=0.95)
        else:
            field = np.ma.masked_invalid(data)
            if tag == 'd':
                field = np.ma.masked_where(dual_mask < 0.5, data)
            cs = ax.contourf(P_grid, Q_grid, field, levels=18, cmap=cmap)
            plt.colorbar(cs, ax=ax, fraction=0.045, pad=0.02)
        ax.contour(P_grid, Q_grid, labels, levels=[0.5], colors='#24323b', linewidths=1.3)
        ax.scatter([350], [-350], marker='*', s=200, color='gold', edgecolors='black', lw=0.8, zorder=10)
        ax.text(0.01, 0.99, tag, transform=ax.transAxes, ha='left', va='top',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='#b8c2c9', linewidth=0.8))
        ax.set_xlabel('Load active power P2 (MW)', fontweight='bold')
        ax.set_ylabel('Load reactive power Q2 (MVAR)', fontweight='bold')
        ax.set_title(title, fontweight='bold')

    fig.suptitle(
        'WB2 Static Security Region and Feasible-Point Internal States',
        fontsize=13,
        fontweight='bold',
        y=0.99,
    )
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
    model_names = ['Baseline', 'Physics-NN', 'SSR-PDNet']

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
        ax.set_ylim((0.85, 1.05))
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
            grid_shape = _infer_grid_shape(X_g)
            if grid_shape is None:
                n_sq = int(np.sqrt(len(y_g)))
                if n_sq * n_sq == len(y_g):
                    grid_shape = (n_sq, n_sq)

            # Try to predict with SSR-PDNet
            from models import SSR_PDNet
            n_bus_map = {'WB2': 2, 'WB5': 5, 'LMBM3': 3, 'case9mod': 9}
            n_bus = n_bus_map.get(case_name, 9)
            input_dim = X_g.shape[1]
            n_feat_dim = {'WB2': 64, 'WB5': 128, 'case9mod': 256}.get(case_name, 128)
            cls_dims = {'WB2': [128,128,64], 'WB5': [256,256,128,64], 'case9mod': [512,512,256,128]}.get(case_name, [256,256,128,64])
            phys_dims = {'WB2': [64,32], 'WB5': [128,64], 'case9mod': [256,128]}.get(case_name, [128,64])

            ssr = SSR_PDNet(input_dim=input_dim, feature_dim=n_feat_dim,
                        classifier_dims=cls_dims, physics_dims=phys_dims, n_bus=n_bus)
            model_files = list(SAVE_DIR.glob(f'{case_name}*ssr*pth'))
            if model_files:
                try:
                    ssr.load_state_dict(torch.load(str(model_files[0]), map_location='cpu', weights_only=True))
                except:
                    ssr.load_state_dict(torch.load(str(model_files[0]), map_location='cpu'))
                ssr.eval()
                if grid_shape is None:
                    ax3.text(0.5, 0.5, 'Grid shape not supported', ha='center', va='center')
                    ax3.set_title(f'{case_name}: SSR-PDNet Prediction', fontweight='bold')
                else:
                    n_y, n_x = grid_shape
                    probs = _predict_probs_batched(ssr, X_g)
                    labels_2d = y_g.reshape(n_y, n_x)
                    probs_2d = probs.reshape(n_y, n_x)

                    xlab, ylab, _, _ = _axis_info(case_name)
                    X_raw = _to_raw_axes(case_name, X_g)
                    x1 = X_raw[:, 0].reshape(n_y, n_x)
                    x2 = X_raw[:, 1].reshape(n_y, n_x)

                    green_red = LinearSegmentedColormap.from_list('gr', [INFEAS_COLOR, FEAS_COLOR])
                    cs = ax3.contourf(x1, x2, probs_2d, levels=30, cmap=green_red, alpha=0.85, vmin=0, vmax=1)
                    ax3.contour(x1, x2, probs_2d, levels=[0.5], colors='black', linewidths=2.0)
                    ax3.contour(x1, x2, labels_2d, levels=[0.5], colors='white', linewidths=1.5, linestyles='--')
                    ax3.plot([], [], 'k-', lw=2, label='SSR-PDNet boundary')
                    ax3.plot([], [], 'w--', lw=1.5, label='True boundary')
                    acc_g = 1 - np.abs((probs_2d > 0.5).astype(float) - labels_2d).mean()
                    ax3.set_title(f'{case_name}: SSR-PDNet Prediction\n(Grid acc={acc_g:.3f})', fontweight='bold')
                    ax3.legend(fontsize=8)
                    ax3.set_xlabel(xlab)
                    ax3.set_ylabel(ylab)
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
        ssr_f1 = results.get('SSR-PDNet', {}).get('f1', 0)
        text = f"Best F1: {best_f1:.4f}\nBest Acc: {best_acc:.4f}\nSSR-PDNet F1: {ssr_f1:.4f}"
        ax4.text(0.5, 0.4, text, ha='center', va='center', fontsize=10,
                transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    fig.suptitle('Static Security Region Characterization — Bukhsh et al. (2013) Test Cases\n'
                 'Comparison: Baseline NN vs Physics-NN vs SSR-PDNet (Proposed)',
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
