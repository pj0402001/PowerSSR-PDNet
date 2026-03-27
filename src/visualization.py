"""
Visualization utilities for power system static security region.
Generates publication-quality figures showing:
1. 2D security region boundary in load space
2. Training curves comparison
3. Decision boundary vs ground truth heatmaps
4. Model confidence maps
5. ROC/PR curves
6. Confusion matrices
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Publication-quality style settings
plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Custom colormaps
INFEAS_COLOR = '#d62728'   # red for infeasible
FEAS_COLOR = '#2ca02c'     # green for feasible
BOUNDARY_COLOR = '#ff7f0e'  # orange for boundary
UNCERTAIN_COLOR = '#1f77b4' # blue for uncertain


def plot_security_region_2d(
    P_grid: np.ndarray,
    Q_grid: np.ndarray,
    labels_true: np.ndarray,
    probs_pred: np.ndarray,
    case_name: str,
    load_idx: Tuple[int, int] = (0, 1),
    p_base: np.ndarray = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 5),
) -> plt.Figure:
    """
    Plot 2D security region projection onto two load dimensions.
    Shows ground truth labels, model predictions, and uncertainty.
    """
    n_pts = int(np.sqrt(len(labels_true)))
    if n_pts ** 2 != len(labels_true):
        # Non-grid data: use scatter plot
        return plot_security_region_scatter(
            P_grid[:, load_idx[0]], P_grid[:, load_idx[1]],
            labels_true, probs_pred, case_name, p_base, save_path, figsize
        )

    P1 = P_grid[:, load_idx[0]].reshape(n_pts, n_pts)
    P2 = P_grid[:, load_idx[1]].reshape(n_pts, n_pts)
    labels_2d = labels_true.reshape(n_pts, n_pts)
    probs_2d = probs_pred.reshape(n_pts, n_pts)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Custom colormaps
    green_red = LinearSegmentedColormap.from_list('gr', [INFEAS_COLOR, FEAS_COLOR])
    uncertainty_cmap = LinearSegmentedColormap.from_list('unc',
        ['white', UNCERTAIN_COLOR, 'navy'])

    # Panel 1: Ground truth labels
    ax = axes[0]
    im = ax.contourf(P1, P2, labels_2d, levels=[-0.5, 0.5, 1.5],
                     colors=[INFEAS_COLOR, FEAS_COLOR], alpha=0.7)
    ax.contour(P1, P2, labels_2d, levels=[0.5], colors=['black'], linewidths=2.0)
    ax.set_xlabel(f'Load {load_idx[0]+1} Active Power P (MW)', fontweight='bold')
    ax.set_ylabel(f'Load {load_idx[1]+1} Active Power P (MW)', fontweight='bold')
    ax.set_title('Ground Truth\n(Static Security Region)', fontweight='bold')
    patches = [mpatches.Patch(color=FEAS_COLOR, label='Feasible (Secure)'),
               mpatches.Patch(color=INFEAS_COLOR, label='Infeasible (Insecure)')]
    ax.legend(handles=patches, loc='upper right', fontsize=9)

    if p_base is not None:
        ax.axvline(p_base[load_idx[0]], color='gray', linestyle='--', alpha=0.6, label='Base load')
        ax.axhline(p_base[load_idx[1]], color='gray', linestyle='--', alpha=0.6)
        ax.scatter([p_base[load_idx[0]]], [p_base[load_idx[1]]],
                   marker='*', s=200, color='gold', zorder=5, label='Base operating point')

    # Panel 2: Model prediction
    ax = axes[1]
    im2 = ax.contourf(P1, P2, probs_2d, levels=50, cmap=green_red, alpha=0.85, vmin=0, vmax=1)
    ax.contour(P1, P2, probs_2d, levels=[0.5], colors=['black'], linewidths=2.0, linestyles='-')
    ax.contour(P1, P2, labels_2d, levels=[0.5], colors=['white'], linewidths=1.5, linestyles='--')
    plt.colorbar(im2, ax=ax, label='P(Feasible)')
    ax.set_xlabel(f'Load {load_idx[0]+1} Active Power P (MW)', fontweight='bold')
    ax.set_ylabel(f'Load {load_idx[1]+1} Active Power P (MW)', fontweight='bold')
    ax.set_title('SSR-PDNet Predicted\nSecurity Region', fontweight='bold')
    ax.plot([], [], 'k-', linewidth=2, label='Predicted boundary')
    ax.plot([], [], 'w--', linewidth=1.5, label='True boundary')
    ax.legend(loc='upper right', fontsize=9)

    if p_base is not None:
        ax.scatter([p_base[load_idx[0]]], [p_base[load_idx[1]]],
                   marker='*', s=200, color='gold', zorder=5)

    # Panel 3: Prediction error (correct/incorrect)
    ax = axes[2]
    preds = (probs_2d > 0.5).astype(float)
    errors = np.abs(preds - labels_2d)

    # Background: feasibility
    ax.contourf(P1, P2, labels_2d, levels=[-0.5, 0.5, 1.5],
                colors=[INFEAS_COLOR, FEAS_COLOR], alpha=0.3)
    # Overlay: errors
    err_cmap = LinearSegmentedColormap.from_list('err', ['white', 'red'])
    im3 = ax.contourf(P1, P2, errors, levels=[0.5, 1.5],
                      colors=['darkred'], alpha=0.6)
    ax.contour(P1, P2, labels_2d, levels=[0.5], colors=['black'], linewidths=1.5)
    ax.contour(P1, P2, probs_2d, levels=[0.5], colors=[BOUNDARY_COLOR], linewidths=1.5)

    accuracy = 1 - errors.mean()
    ax.set_xlabel(f'Load {load_idx[0]+1} Active Power P (MW)', fontweight='bold')
    ax.set_ylabel(f'Load {load_idx[1]+1} Active Power P (MW)', fontweight='bold')
    ax.set_title(f'Prediction Errors\n(Accuracy: {accuracy:.3f})', fontweight='bold')

    error_patch = mpatches.Patch(color='darkred', alpha=0.7, label='Misclassified')
    ax.legend(handles=[error_patch], loc='upper right', fontsize=9)

    if p_base is not None:
        ax.scatter([p_base[load_idx[0]]], [p_base[load_idx[1]]],
                   marker='*', s=200, color='gold', zorder=5)

    fig.suptitle(f'Static Security Region — {case_name.upper()}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_security_region_scatter(
    P1: np.ndarray,
    P2: np.ndarray,
    labels_true: np.ndarray,
    probs_pred: np.ndarray,
    case_name: str,
    p_base: np.ndarray = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 5),
) -> plt.Figure:
    """Scatter plot version of security region visualization."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    colors_true = [FEAS_COLOR if l > 0.5 else INFEAS_COLOR for l in labels_true]
    preds = (probs_pred > 0.5).astype(float)
    correct = (preds == labels_true)

    # Ground truth
    ax = axes[0]
    ax.scatter(P1[labels_true == 0], P2[labels_true == 0],
               c=INFEAS_COLOR, s=8, alpha=0.6, label='Infeasible')
    ax.scatter(P1[labels_true == 1], P2[labels_true == 1],
               c=FEAS_COLOR, s=8, alpha=0.6, label='Feasible')
    ax.set_title('Ground Truth\n(Static Security Region)', fontweight='bold')
    ax.set_xlabel('Load P1 (MW)', fontweight='bold')
    ax.set_ylabel('Load P2 (MW)', fontweight='bold')
    ax.legend(fontsize=9)

    # Model predictions (colored by probability)
    ax = axes[1]
    green_red = LinearSegmentedColormap.from_list('gr', [INFEAS_COLOR, FEAS_COLOR])
    scatter = ax.scatter(P1, P2, c=probs_pred, cmap=green_red, s=8, alpha=0.7, vmin=0, vmax=1)
    plt.colorbar(scatter, ax=ax, label='P(Feasible)')
    ax.set_title('SSR-PDNet Predicted Probability', fontweight='bold')
    ax.set_xlabel('Load P1 (MW)', fontweight='bold')
    ax.set_ylabel('Load P2 (MW)', fontweight='bold')

    # Errors
    ax = axes[2]
    ax.scatter(P1[correct], P2[correct], c='steelblue', s=5, alpha=0.4, label='Correct')
    ax.scatter(P1[~correct], P2[~correct], c='red', s=20, alpha=0.8,
               marker='x', label='Error', zorder=5)
    ax.set_title(f'Prediction Errors\nAcc={correct.mean():.3f}', fontweight='bold')
    ax.set_xlabel('Load P1 (MW)', fontweight='bold')
    ax.set_ylabel('Load P2 (MW)', fontweight='bold')
    ax.legend(fontsize=9)

    fig.suptitle(f'Static Security Region — {case_name.upper()}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def plot_training_curves(
    histories: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 8),
) -> plt.Figure:
    """Compare training curves of multiple models."""
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    metrics = [
        ('train_loss', 'Training Loss', axes[0, 0]),
        ('val_loss', 'Validation Loss', axes[0, 1]),
        ('val_f1', 'Validation F1 Score', axes[0, 2]),
        ('val_acc', 'Validation Accuracy', axes[1, 0]),
    ]

    for metric, title, ax in metrics:
        for i, (name, hist) in enumerate(histories.items()):
            if metric in hist:
                vals = hist[metric]
                epochs = range(1, len(vals) + 1)
                ax.plot(epochs, vals, color=colors[i % len(colors)],
                        label=name, linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9)

    # SSR-PDNet specific: loss components
    if 'SSR-PDNet' in histories and 'loss_focal' in histories['SSR-PDNet']:
        ax = axes[1, 1]
        hist = histories['SSR-PDNet']
        for comp, color, label in [
            ('loss_focal', '#1f77b4', 'Focal Loss'),
            ('loss_physics', '#ff7f0e', 'Physics Loss'),
            ('loss_contrastive', '#2ca02c', 'Contrastive Loss'),
        ]:
            if comp in hist:
                ax.plot(hist[comp], color=color, label=label, linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('SSR-PDNet Loss Components', fontweight='bold')
        ax.legend(fontsize=9)

    # Lagrange multiplier evolution
    if 'SSR-PDNet' in histories and 'lambda_v' in histories['SSR-PDNet']:
        ax = axes[1, 2]
        hist = histories['SSR-PDNet']
        ax.plot(hist['lambda_v'], color='#d62728', label='λ_v (voltage)', linewidth=1.5)
        if 'lambda_l' in hist:
            ax.plot(hist['lambda_l'], color='#9467bd', label='λ_l (line)', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Lagrange Multiplier Value')
        ax.set_title('Dual Variable Evolution', fontweight='bold')
        ax.legend(fontsize=9)
    else:
        axes[1, 2].set_visible(False)

    plt.suptitle('Training Dynamics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def plot_roc_pr_curves(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Plot ROC and PR curves for all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    linestyles = ['-', '--', '-.', ':']

    for i, (name, res) in enumerate(results.items()):
        probs = res['probs']
        labels = res['labels']
        c = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]

        # ROC curve
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=c, linestyle=ls, linewidth=2,
                 label=f'{name} (AUC={roc_auc:.3f})')

        # PR curve
        prec, rec, _ = precision_recall_curve(labels, probs)
        pr_auc = auc(rec, prec)
        ax2.plot(rec, prec, color=c, linestyle=ls, linewidth=2,
                 label=f'{name} (AUC={pr_auc:.3f})')

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax1.set_xlabel('False Positive Rate (FPR)', fontweight='bold')
    ax1.set_ylabel('True Positive Rate (TPR)', fontweight='bold')
    ax1.set_title('ROC Curves', fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.02])

    ax2.set_xlabel('Recall', fontweight='bold')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_title('Precision-Recall Curves', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.02])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def plot_comparison_bar(
    results: Dict[str, Dict],
    metrics: List[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Bar chart comparison of models across metrics."""
    if metrics is None:
        metrics = ['acc', 'f1', 'prec', 'rec', 'spec']

    metric_names = {
        'acc': 'Accuracy', 'f1': 'F1 Score',
        'prec': 'Precision', 'rec': 'Recall',
        'spec': 'Specificity',
    }

    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.7 / n_models

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, ax = plt.subplots(figsize=figsize)

    for i, name in enumerate(model_names):
        vals = [results[name].get(m, 0) for m in metrics]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=name,
                      color=colors[i % len(colors)], alpha=0.85, edgecolor='white')

        # Value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([metric_names.get(m, m) for m in metrics], fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_ylim([0, 1.08])
    ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def plot_confusion_matrix(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 4),
) -> plt.Figure:
    """Confusion matrices for all models."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    cmap = sns.color_palette("Blues", as_cmap=True)

    for ax, (name, res) in zip(axes, results.items()):
        probs = res['probs']
        labels = res['labels'].astype(int)
        th = res.get('best_threshold', 0.5)
        preds = (probs > th).astype(int)

        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                    xticklabels=['Pred: Infeas.', 'Pred: Feas.'],
                    yticklabels=['True: Infeas.', 'True: Feas.'],
                    linewidths=0.5, annot_kws={'size': 12})
        ax.set_title(f'{name}\n(th={th:.2f})', fontweight='bold')

    plt.suptitle('Confusion Matrices', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def plot_feasibility_rate_vs_load(
    case_names: List[str],
    feasibility_rates: List[float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot feasibility rates across different test cases."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [FEAS_COLOR if r > 0.5 else INFEAS_COLOR for r in feasibility_rates]
    bars = ax.bar(case_names, feasibility_rates, color=colors, alpha=0.8, edgecolor='white')

    for bar, rate in zip(bars, feasibility_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Feasibility Rate', fontweight='bold')
    ax.set_xlabel('Test Case', fontweight='bold')
    ax.set_title('Dataset Feasibility Rates by Test Case', fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.6, label='50% baseline')
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig
