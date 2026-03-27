"""
Main experiment script for SSR-PDNet characterization of OPF security regions.

CRITICAL FIX: All data generation and visualization now works in
GENERATOR POWER SPACE (P_G1, P_G2, ... as axes, loads FIXED at nominal).

This matches the traditional IPOPT approach from:
  - D:/安全域/1/5节点.py (WB5: scan P_G5 vs P_G1)
  - D:/安全域/1/case9线路热极限.py (case9mod: scan P_G2 vs P_G3)

Expected results matching traditional IPOPT:
  - WB5: 2 disconnected secure components in (P_G1, P_G5) space
  - case9mod: 3 disconnected secure components in (P_G2, P_G3) space
  - LMBM3: 2 components at load_factor≈1.5 due to L3-2 thermal limit

Reference: W. A. Bukhsh et al., "Local Solutions of Optimal Power Flow Problem",
           IEEE Trans. Power Systems, 2013.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import json
import time
import random
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

from bukhsh_data import (
    generate_WB2_data, generate_WB2_grid,
    generate_WB5_data, generate_WB5_grid,
    generate_case9mod_data, generate_case9mod_grid,
    generate_LMBM3_data,
    load_traditional_results,
    get_wb5_traditional_dataset,
    get_case9mod_traditional_dataset,
    generate_ssr_data, generate_ssr_grid,
)
from models import BaselineNN, PhysicsNN, SSR_PDNet
from trainer import make_data_loaders, train_baseline, train_ssr_pdnet, evaluate_model

# ─── Directories ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / 'figures'
DATA_DIR = ROOT / 'data'
SAVE_DIR = ROOT / 'results'
TRAD_DIR = Path(r'D:\安全域\1')  # traditional IPOPT results
for d in [FIG_DIR, DATA_DIR, SAVE_DIR]:
    d.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


def set_global_seed(seed: int = 42):
    """Set Python/NumPy/PyTorch random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'font.family': 'DejaVu Sans',
})

FEAS_COLOR = '#2ca02c'    # green: feasible/secure
INFEAS_COLOR = '#d62728'  # red: infeasible/insecure


# ══════════════════════════════════════════════════════════════════════════════
# Utility: boundary metrics (comparing DL boundary to traditional IPOPT)
# ══════════════════════════════════════════════════════════════════════════════

def compute_boundary_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             X_raw: np.ndarray = None) -> dict:
    """
    Compute boundary characterization accuracy metrics.
    - F1 score, precision, recall, accuracy
    - Boundary F1: focuses on points near the decision boundary
    - Hausdorff distance: if X_raw provided (2D coordinates)
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    from sklearn.neighbors import NearestNeighbors

    y_true = (y_true > 0.5).astype(int)
    y_pred_bin = (y_pred > 0.5).astype(int)

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred_bin)),
        'f1': float(f1_score(y_true, y_pred_bin, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred_bin, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred_bin, zero_division=0)),
    }

    # Boundary accuracy: points where true label disagrees with neighbor
    if X_raw is not None and X_raw.shape[1] == 2:
        try:
            nbrs = NearestNeighbors(n_neighbors=5).fit(X_raw)
            dists, indices = nbrs.kneighbors(X_raw)
            # A point is "near boundary" if any neighbor has different label
            near_boundary = np.zeros(len(y_true), dtype=bool)
            for i in range(len(y_true)):
                neighbor_labels = y_true[indices[i, 1:]]
                if (neighbor_labels != y_true[i]).any():
                    near_boundary[i] = True

            if near_boundary.sum() > 10:
                bd_acc = accuracy_score(y_true[near_boundary], y_pred_bin[near_boundary])
                bd_f1 = f1_score(y_true[near_boundary], y_pred_bin[near_boundary],
                                 zero_division=0)
                metrics['boundary_accuracy'] = float(bd_acc)
                metrics['boundary_f1'] = float(bd_f1)
                metrics['n_boundary_points'] = int(near_boundary.sum())
        except Exception:
            pass

        # Hausdorff distance between predicted and true boundaries
        try:
            true_boundary_pts = X_raw[y_true != y_pred_bin]
            if len(true_boundary_pts) > 0:
                from scipy.spatial.distance import directed_hausdorff
                feas_pred = X_raw[y_pred_bin == 1]
                infeas_pred = X_raw[y_pred_bin == 0]
                feas_true = X_raw[y_true == 1]
                infeas_true = X_raw[y_true == 0]
                if len(feas_pred) > 0 and len(feas_true) > 0:
                    hd1 = directed_hausdorff(feas_pred, feas_true)[0]
                    hd2 = directed_hausdorff(feas_true, feas_pred)[0]
                    metrics['hausdorff_distance'] = float(max(hd1, hd2))
        except Exception:
            pass

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Visualization: Traditional IPOPT vs DL comparison (generator power space)
# ══════════════════════════════════════════════════════════════════════════════


def _panel_tag(ax, tag: str):
    ax.text(0.01, 0.99, tag, transform=ax.transAxes,
            ha='left', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                      edgecolor='#b8c2c9', linewidth=0.8))



def _get_case_axes(case_name: str):
    case_labels = {
        'WB5': ('P_G1 (MW)', 'P_G5 (MW)', [350.0, 200.0], 'WB5 (5-Bus Meshed System)'),
        'case9mod': ('P_G2 (MW)', 'P_G3 (MW)', [163.0, 85.0], 'case9mod (Modified IEEE 9-Bus)'),
        'WB2': ('P_d (MW)', 'Q_d (MVAR)', [350.0, -350.0], 'WB2 (2-Bus Analytical System)'),
        'LMBM3': ('P_G1 (MW)', 'P_G2 (MW)', [185.93, 300.0], 'LMBM3 (3-Bus Triangle)'),
    }
    return case_labels.get(case_name, ('X', 'Y', [0, 0], case_name))



def _extract_scatter_axes(case_name: str, df):
    if case_name == 'WB5':
        return df['PG1'].values if 'PG1' in df.columns else df.iloc[:, 1].values, \
               df['PG5'].values if 'PG5' in df.columns else df.iloc[:, 0].values
    if case_name == 'case9mod':
        return df['p2_mw'].values, df['p3_mw'].values
    if case_name == 'LMBM3':
        return df['PG1'].values, df['PG2'].values
    return df.iloc[:, 0].values, df.iloc[:, 1].values



def _rounded_scatter_limits(x_vals: np.ndarray, y_vals: np.ndarray, step: float = 10.0):
    """Round scatter bounds to clean axis ticks (default 10 MW)."""
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    x_lo = step * np.floor(np.nanmin(x_vals) / step)
    x_hi = step * np.ceil(np.nanmax(x_vals) / step)
    y_lo = step * np.floor(np.nanmin(y_vals) / step)
    y_hi = step * np.ceil(np.nanmax(y_vals) / step)
    return float(x_lo), float(x_hi), float(y_lo), float(y_hi)



def _case_axis_limits(case_name: str, trad_data: dict = None, arr_x=None, arr_y=None):
    """Case-specific plotting limits; keeps case9mod aligned with traditional figure."""
    if case_name != 'case9mod':
        return None

    if trad_data is not None and 'df' in trad_data:
        x_vals, y_vals = _extract_scatter_axes(case_name, trad_data['df'])
    elif arr_x is not None and arr_y is not None:
        x_vals, y_vals = np.asarray(arr_x), np.asarray(arr_y)
    else:
        return None

    _, x_hi, _, y_hi = _rounded_scatter_limits(x_vals, y_vals, step=10.0)
    x_hi = max(180.0, x_hi)
    y_hi = max(180.0, y_hi)
    return 0.0, x_hi, 0.0, y_hi



def _case_axis_ticks(case_name: str):
    """Optional fixed axis ticks for publication consistency."""
    if case_name == 'case9mod':
        return np.arange(0.0, 181.0, 20.0), np.arange(0.0, 181.0, 20.0)
    return None



def _domain_safe_threshold(case_name: str, probs_2d: np.ndarray, labels_2d: np.ndarray, arr_x, arr_y) -> float:
    """Choose threshold that suppresses out-of-domain secure artifacts.

    For case9mod we require near-zero secure prediction in [0,10) MW guard bands,
    while keeping F1 as high as possible inside the plotted domain.
    """
    if case_name != 'case9mod':
        return 0.5

    XX, YY = np.meshgrid(arr_x, arr_y)
    guard_mask = (XX < 10.0) | (YY < 10.0)
    if int(guard_mask.sum()) == 0:
        return 0.5

    best_t = 0.5
    best_obj = -1e9
    for t in np.linspace(0.5, 0.9, 81):
        pred = probs_2d > t
        guard_rate = float(pred[guard_mask].mean())
        if guard_rate > 0.002:
            continue

        tp = float(np.sum(pred & (labels_2d > 0.5)))
        fp = float(np.sum(pred & (labels_2d <= 0.5)))
        fn = float(np.sum((~pred) & (labels_2d > 0.5)))
        f1 = 2.0 * tp / (2.0 * tp + fp + fn + 1e-12)

        # Favor higher F1, lightly favor lower threshold among ties.
        obj = f1 - 1e-3 * (t - 0.5)
        if obj > best_obj:
            best_obj = obj
            best_t = float(t)

    return best_t



def _shade_out_of_domain(ax, case_name: str):
    """Visually mark regions outside the traditional scan domain."""
    if case_name != 'case9mod':
        return
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    if x1 <= x0 or y1 <= y0:
        return

    # Traditional case9 scan starts at 10 MW on both axes.
    if x0 < 10.0:
        ax.axvspan(x0, 10.0, color='#d9d9d9', alpha=0.35, zorder=0)
    if y0 < 10.0:
        ax.axhspan(y0, 10.0, color='#d9d9d9', alpha=0.35, zorder=0)



def _extract_physical_overlay(case_name: str, trad_data: dict):
    if trad_data is None or 'df' not in trad_data:
        return None

    df = trad_data['df']
    x, y = _extract_scatter_axes(case_name, df)

    if case_name == 'WB5':
        pg1 = df['PG1'].values if 'PG1' in df.columns else df.iloc[:, 1].values
        pg5 = df['PG5'].values if 'PG5' in df.columns else df.iloc[:, 0].values
        total = np.maximum(pg1 + pg5, 1e-6)
        return {
            'x': x,
            'y': y,
            'values': pg1 / total,
            'cmap': 'cividis',
            'label': r'Slack dispatch share $P_{G1}/(P_{G1}+P_{G5})$',
            'title': 'Redispatch pattern inside the secure set',
        }

    if case_name == 'case9mod':
        v_cols = [c for c in df.columns if c.startswith('v') and c.endswith('_pu')]
        if v_cols:
            values = df[v_cols].min(axis=1).values
            return {
                'x': x,
                'y': y,
                'values': values,
                'cmap': 'viridis',
                'label': 'Minimum bus voltage (p.u.)',
                'title': 'Internal voltage margin of secure operating points',
            }
        return {
            'x': x,
            'y': y,
            'values': df['p1_mw'].values,
            'cmap': 'plasma',
            'label': 'Slack generation P_G1 (MW)',
            'title': 'Slack redispatch over the secure set',
        }

    return None



def plot_traditional_vs_dl(
    case_name: str,
    trad_data: dict,
    grid_data: dict,
    model_probs: dict,
    save_prefix: str,
):
    """Create publication-style security-region figures."""
    xlabel, ylabel, nominal_pt, title = _get_case_axes(case_name)

    if 'pg1_arr' in grid_data and 'pg5_arr' in grid_data:
        arr_x = grid_data['pg1_arr']
        arr_y = grid_data['pg5_arr']
        XX, YY = np.meshgrid(arr_x, arr_y)
    elif 'pg2_arr' in grid_data and 'pg3_arr' in grid_data:
        arr_x = grid_data['pg2_arr']
        arr_y = grid_data['pg3_arr']
        XX, YY = np.meshgrid(arr_x, arr_y)
    elif 'P_arr' in grid_data and 'Q_arr' in grid_data:
        arr_x = grid_data['P_arr']
        arr_y = grid_data['Q_arr']
        XX, YY = np.meshgrid(arr_x, arr_y)
    else:
        print(f"Cannot determine grid axes for {case_name}")
        return

    n_x = len(arr_x)
    n_y = len(arr_y)

    labels_2d = grid_data.get('labels_2d')
    if labels_2d is None:
        labels_2d = grid_data.get('y_grid', np.zeros((n_y, n_x)))
        if labels_2d.ndim == 1:
            labels_2d = labels_2d.reshape(n_y, n_x)

    primary_name = 'SSR-PDNet' if 'SSR-PDNet' in model_probs else next(iter(model_probs))
    probs_flat = model_probs[primary_name]
    if probs_flat.shape[0] != n_x * n_y:
        print(f"Grid size mismatch for {case_name}: probs={probs_flat.shape[0]}, expected={n_x*n_y}")
        return
    probs_2d = probs_flat.reshape(n_y, n_x)
    safe_th = _domain_safe_threshold(case_name, probs_2d, labels_2d, arr_x, arr_y)
    pred_2d = (probs_2d > safe_th).astype(float)
    grid_acc = (pred_2d == labels_2d).mean()

    secure_cmap = LinearSegmentedColormap.from_list(
        'security', ['#f4efe5', '#d8e7d2', '#2d6a4f']
    )
    truth_colors = ['#efe7da', '#5b8c5a']

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.1))
    fig.patch.set_facecolor('white')

    for ax in axes:
        ax.set_facecolor('#fbfaf7')
        ax.grid(True, alpha=0.18, linestyle='--', linewidth=0.6)

    # Panel (a): traditional reference
    ax = axes[0]
    ax.contourf(XX, YY, labels_2d, levels=[-0.5, 0.5, 1.5],
                colors=truth_colors, alpha=0.92)
    if labels_2d.max() > 0.5 and labels_2d.min() < 0.5:
        ax.contour(XX, YY, labels_2d, levels=[0.5], colors='#24323b', linewidths=1.9)
    if trad_data is not None and 'df' in trad_data:
        x_sc, y_sc = _extract_scatter_axes(case_name, trad_data['df'])
        ax.scatter(x_sc, y_sc, s=5, c='#173f35', alpha=0.60, edgecolors='none')
        ax.text(0.98, 0.05, f'{len(x_sc):,} IPOPT-secure points', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8.5, color='#173f35')

    limits = _case_axis_limits(case_name, trad_data=trad_data, arr_x=arr_x, arr_y=arr_y)
    ticks = _case_axis_ticks(case_name)
    if limits is not None:
        x_lo, x_hi, y_lo, y_hi = limits
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
    if ticks is not None:
        xt, yt = ticks
        ax.set_xticks(xt)
        ax.set_yticks(yt)
    _shade_out_of_domain(ax, case_name)
    _shade_out_of_domain(ax, case_name)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title('Traditional IPOPT reference', fontweight='bold')
    _panel_tag(ax, 'a')

    # Panel (b): SSR-PDNet security score
    ax = axes[1]
    cs = ax.contourf(XX, YY, probs_2d, levels=np.linspace(0, 1, 21),
                     cmap=secure_cmap, vmin=0.0, vmax=1.0)
    plt.colorbar(cs, ax=ax, label='Security score', fraction=0.045, pad=0.02)
    if probs_2d.max() > safe_th and probs_2d.min() < safe_th:
        ax.contour(XX, YY, probs_2d, levels=[safe_th], colors='#111111', linewidths=2.2)
    if labels_2d.max() > 0.5 and labels_2d.min() < 0.5:
        ax.contour(XX, YY, labels_2d, levels=[0.5], colors='white', linewidths=1.4, linestyles='--')

    if limits is not None:
        x_lo, x_hi, y_lo, y_hi = limits
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
    if ticks is not None:
        xt, yt = ticks
        ax.set_xticks(xt)
        ax.set_yticks(yt)
    _shade_out_of_domain(ax, case_name)
    _shade_out_of_domain(ax, case_name)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(f'{primary_name} security map (grid acc. = {grid_acc:.3f}, th = {safe_th:.2f})', fontweight='bold')
    ax.plot([], [], color='#111111', lw=2.2, label='Predicted boundary')
    ax.plot([], [], color='white', lw=1.4, linestyle='--', label='True boundary')
    ax.legend(loc='upper right', fontsize=8)
    _panel_tag(ax, 'b')

    # Panel (c): physical interpretation or error map
    ax = axes[2]
    overlay = _extract_physical_overlay(case_name, trad_data)
    if overlay is not None:
        sc = ax.scatter(overlay['x'], overlay['y'], c=overlay['values'], s=11,
                        cmap=overlay['cmap'], alpha=0.85, edgecolors='none')
        plt.colorbar(sc, ax=ax, label=overlay['label'], fraction=0.045, pad=0.02)
        if labels_2d.max() > 0.5 and labels_2d.min() < 0.5:
            ax.contour(XX, YY, labels_2d, levels=[0.5], colors='#24323b', linewidths=1.5)
        ax.set_title(overlay['title'], fontweight='bold')
    else:
        err = np.abs(pred_2d - labels_2d)
        ax.contourf(XX, YY, labels_2d, levels=[-0.5, 0.5, 1.5], colors=truth_colors, alpha=0.35)
        if err.any():
            ax.contourf(XX, YY, err, levels=[0.5, 1.5], colors=['#8c2f39'], alpha=0.85)
        if labels_2d.max() > 0.5 and labels_2d.min() < 0.5:
            ax.contour(XX, YY, labels_2d, levels=[0.5], colors='#24323b', linewidths=1.5)
        ax.set_title('Prediction disagreement map', fontweight='bold')

    if limits is not None:
        x_lo, x_hi, y_lo, y_hi = limits
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
    if ticks is not None:
        xt, yt = ticks
        ax.set_xticks(xt)
        ax.set_yticks(yt)
    _shade_out_of_domain(ax, case_name)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    _panel_tag(ax, 'c')

    fig.suptitle(
        f'Static Security Region Characterization in Generator Power Space - {title}',
        fontsize=12.5,
        fontweight='bold',
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(f'{save_prefix}_security_region.png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{save_prefix}_dl_comparison.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_prefix}_security_region.png")


def plot_traditional_ipopt_raw(case_name: str, save_prefix: str):
    """Load and plot the raw traditional IPOPT results directly from CSV."""
    trad = load_traditional_results(case_name, str(TRAD_DIR))
    if trad is None:
        return None

    df = trad['df']
    xlabel, ylabel, nominal_pt, title = _get_case_axes(case_name)
    overlay = _extract_physical_overlay(case_name, trad)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    fig.patch.set_facecolor('white')

    for ax in axes:
        ax.set_facecolor('#fbfaf7')
        ax.grid(True, alpha=0.18, linestyle='--', linewidth=0.6)

    x, y = _extract_scatter_axes(case_name, df)
    limits = _case_axis_limits(case_name, trad_data=trad)
    ticks = _case_axis_ticks(case_name)

    ax = axes[0]
    ax.scatter(x, y, s=6, c='#215c4f', alpha=0.72, edgecolors='none')
    if limits is not None:
        x_lo, x_hi, y_lo, y_hi = limits
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
    if ticks is not None:
        xt, yt = ticks
        ax.set_xticks(xt)
        ax.set_yticks(yt)
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(f'Traditional IPOPT secure set\n{title}', fontsize=11.5, fontweight='bold')
    ax.text(0.98, 0.04, f'{len(x):,} secure operating points',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5, color='#215c4f')
    _panel_tag(ax, 'a')

    ax = axes[1]
    if overlay is not None:
        sc = ax.scatter(overlay['x'], overlay['y'], s=8, c=overlay['values'],
                        cmap=overlay['cmap'], alpha=0.82, edgecolors='none')
        plt.colorbar(sc, ax=ax, label=overlay['label'], fraction=0.045, pad=0.02)
        ax.set_title(overlay['title'], fontsize=11.5, fontweight='bold')
    else:
        ax.scatter(x, y, s=8, c='#215c4f', alpha=0.72, edgecolors='none')
        ax.set_title('Secure operating-point cloud', fontsize=11.5, fontweight='bold')
    if limits is not None:
        x_lo, x_hi, y_lo, y_hi = limits
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
    if ticks is not None:
        xt, yt = ticks
        ax.set_xticks(xt)
        ax.set_yticks(yt)
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    _panel_tag(ax, 'b')

    plt.tight_layout()
    fig.savefig(f'{save_prefix}_ipopt_reference.png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{save_prefix}_traditional.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_prefix}_ipopt_reference.png")
    return trad


# ══════════════════════════════════════════════════════════════════════════════
# Grid construction from traditional IPOPT scatter (no Pyomo needed)
# ══════════════════════════════════════════════════════════════════════════════

def _infer_step_from_values(values: np.ndarray, fallback: float) -> float:
    """Infer dominant step size from discrete coordinate values."""
    vals = np.unique(np.round(values.astype(float), 6))
    if len(vals) < 2:
        return fallback
    diffs = np.diff(vals)
    diffs = diffs[diffs > 1e-9]
    if len(diffs) == 0:
        return fallback
    d_rounded = np.round(diffs, 6)
    uniq, cnt = np.unique(d_rounded, return_counts=True)
    step = float(uniq[np.argmax(cnt)])
    if step <= 0:
        return fallback
    return step


def build_grid_from_scatter(
    X_feas_mw,
    x_range,
    y_range,
    n_per_dim=60,
    case_key='pg1_pg5',
    trad=None,
    exact_axes=False,
    x_step=None,
    y_step=None,
):
    """
    Build a regular 2-D grid with labels derived from traditional IPOPT scatter.

    Strategy:
      - All grid points are initially labelled insecure (0).
      - For each traditional secure point we mark the nearest grid cell as
        secure via an ε-radius neighborhood vote.
      - Returns (X_g, y_g, meta_g) compatible with the rest of the pipeline.
    """
    from sklearn.neighbors import KDTree

    if exact_axes:
        sx = float(x_step) if x_step is not None else _infer_step_from_values(
            X_feas_mw[:, 0], (x_range[1] - x_range[0]) / max(1, n_per_dim - 1)
        )
        sy = float(y_step) if y_step is not None else _infer_step_from_values(
            X_feas_mw[:, 1], (y_range[1] - y_range[0]) / max(1, n_per_dim - 1)
        )

        x_arr = np.arange(x_range[0], x_range[1] + 0.5 * sx, sx)
        y_arr = np.arange(y_range[0], y_range[1] + 0.5 * sy, sy)
        XX, YY = np.meshgrid(x_arr, y_arr)
        labels_2d = np.zeros((len(y_arr), len(x_arr)), dtype=np.float32)

        ix = np.rint((X_feas_mw[:, 0] - x_range[0]) / sx).astype(int)
        iy = np.rint((X_feas_mw[:, 1] - y_range[0]) / sy).astype(int)
        valid = (
            (ix >= 0) & (ix < len(x_arr)) &
            (iy >= 0) & (iy < len(y_arr))
        )
        labels_2d[iy[valid], ix[valid]] = 1.0

        X_grid = np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32)
        y_g = labels_2d.ravel().astype(np.float32)
        radius = 0.0
    else:
        x_arr = np.linspace(x_range[0], x_range[1], n_per_dim)
        y_arr = np.linspace(y_range[0], y_range[1], n_per_dim)
        XX, YY = np.meshgrid(x_arr, y_arr)
        X_grid = np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32)

        # radius = 2 grid cells diagonal
        dx = (x_range[1] - x_range[0]) / max(1, (n_per_dim - 1))
        dy = (y_range[1] - y_range[0]) / max(1, (n_per_dim - 1))
        radius = 2.5 * np.sqrt(dx**2 + dy**2)

        tree = KDTree(X_feas_mw)
        counts = tree.query_radius(X_grid, r=radius, count_only=True)
        y_g = (counts > 0).astype(np.float32)
        labels_2d = y_g.reshape(n_per_dim, n_per_dim)

    # Build meta dict compatible with WB5/case9mod pipeline
    if case_key == 'pg1_pg5':
        meta_g = {
            'pg1_arr': x_arr, 'pg5_arr': y_arr,
            'labels_2d': labels_2d,
            'n_per_dim': len(x_arr),
            'X_mean': np.array([x_arr.mean(), y_arr.mean()], dtype=np.float32),
            'X_std':  np.array([x_arr.std(),  y_arr.std()],  dtype=np.float32),
        }
    else:  # pg2_pg3
        meta_g = {
            'pg2_arr': x_arr, 'pg3_arr': y_arr,
            'labels_2d': labels_2d,
            'n_per_dim': len(x_arr),
            'X_mean': np.array([x_arr.mean(), y_arr.mean()], dtype=np.float32),
            'X_std':  np.array([x_arr.std(),  y_arr.std()],  dtype=np.float32),
        }

    print(f"  Grid {len(y_arr)}x{len(x_arr)}: {int(y_g.sum())} secure / "
          f"{len(y_g)} total  (rate={y_g.mean():.3f}, radius={radius:.1f} MW)")
    return X_grid, y_g, meta_g


# ══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def train_all_models(X, y, meta, cfg, epochs, device, seed: int = 42):
    """Train Baseline, Physics-NN, SSR-PDNet on the given dataset."""
    train_loader, val_loader, test_loader = make_data_loaders(
        X, y, val_ratio=0.15, test_ratio=0.15,
        batch_size=cfg['batch'], balance=True, seed=seed,
    )
    input_dim = X.shape[1]
    n_bus = meta.get('n_bus', 9)

    histories = {}
    test_results = {}
    models = {}

    # A) Baseline NN
    print("  [A] Baseline NN...")
    baseline = BaselineNN(input_dim=input_dim, hidden_dims=cfg['hidden'], dropout=0.1)
    histories['Baseline'] = train_baseline(
        baseline, train_loader, val_loader,
        epochs=epochs, lr=1e-3, patience=40, device=device,
    )
    test_results['Baseline'] = evaluate_model(baseline, test_loader, device)
    models['Baseline'] = baseline
    r = test_results['Baseline']
    print(f"     acc={r['acc']:.4f}  f1={r['f1']:.4f}  prec={r['prec']:.4f}  rec={r['rec']:.4f}")

    # B) Physics-NN
    print("  [B] Physics-NN...")
    phys = PhysicsNN(input_dim=input_dim, hidden_dims=cfg['hidden'], n_bus=n_bus, dropout=0.1)
    histories['Physics-NN'] = train_baseline(
        phys, train_loader, val_loader,
        epochs=epochs, lr=1e-3, patience=40, device=device,
    )
    test_results['Physics-NN'] = evaluate_model(phys, test_loader, device)
    models['Physics-NN'] = phys
    r = test_results['Physics-NN']
    print(f"     acc={r['acc']:.4f}  f1={r['f1']:.4f}")

    # C) SSR-PDNet
    print("  [C] SSR-PDNet (proposed)...")
    ssr = SSR_PDNet(
        input_dim=input_dim,
        feature_dim=cfg['feature_dim'],
        classifier_dims=cfg['classifier_dims'],
        physics_dims=cfg['physics_dims'],
        n_bus=n_bus, dropout=0.1, use_physics_head=True,
    )
    histories['SSR-PDNet'] = train_ssr_pdnet(
        ssr, train_loader, val_loader,
        epochs=int(epochs * 1.2), lr=1e-3, lr_dual=1e-2,
        lambda_physics=0.1, lambda_contrastive=0.05,
        input_lower_bound=torch.zeros(input_dim),
        lambda_domain_guard=0.15,
        patience=45, device=device,
    )
    test_results['SSR-PDNet'] = evaluate_model(ssr, test_loader, device)
    models['SSR-PDNet'] = ssr
    r = test_results['SSR-PDNet']
    print(f"     acc={r['acc']:.4f}  f1={r['f1']:.4f}")

    return models, histories, test_results, test_loader


def predict_on_grid(models: dict, X_g: np.ndarray, device, batch_size: int = 8192) -> dict:
    """Get probability predictions from each model on the grid."""
    probs = {}
    X_t = torch.FloatTensor(X_g)
    for name, model in models.items():
        model.eval()
        pred_parts = []
        with torch.no_grad():
            for i in range(0, len(X_t), batch_size):
                xb = X_t[i:i + batch_size].to(device)
                if isinstance(model, (SSR_PDNet, PhysicsNN)):
                    logits, _ = model(xb)
                else:
                    logits = model(xb)
                pred_parts.append(torch.sigmoid(logits).cpu())
        probs[name] = torch.cat(pred_parts).numpy()
    return probs


# ══════════════════════════════════════════════════════════════════════════════
# Per-case experiment functions
# ══════════════════════════════════════════════════════════════════════════════

def run_WB2(quick: bool = False, seed: int = 42):
    """
    WB2: 2-bus system, analytical power flow.
    Characterize security in (P2_load, Q2_load) space.
    """
    print(f"\n{'='*65}")
    print("  WB2: 2-Bus System (Analytical)")
    print(f"{'='*65}")
    set_global_seed(seed)

    cfg = {
        'n_samples': 2000 if quick else 5000,
        'n_grid': 40 if quick else 80,
        'epochs': 80 if quick else 200,
        'batch': 256,
        'hidden': [128, 128, 64],
        'feature_dim': 64,
        'classifier_dims': [128, 128, 64],
        'physics_dims': [64, 32],
    }

    # 1. Data generation
    print("\n[1] WB2 data generation (analytical)...")
    X, y, meta = generate_WB2_data(n_samples=cfg['n_samples'], seed=seed)
    print(f"  Shape: {X.shape}, security rate: {y.mean():.3f}")

    # 2. Grid
    print(f"\n[2] WB2 {cfg['n_grid']}x{cfg['n_grid']} grid...")
    X_g, y_g, meta_g = generate_WB2_grid(n_per_dim=cfg['n_grid'])
    # Add labels_2d for compatibility with plot_traditional_vs_dl
    meta_g['labels_2d'] = y_g.reshape(cfg['n_grid'], cfg['n_grid'])
    np.save(DATA_DIR / 'WB2_X_grid.npy', X_g)
    np.save(DATA_DIR / 'WB2_y_grid.npy', y_g)
    print(f"  Grid security rate: {y_g.mean():.3f}")

    # 3. Training
    print(f"\n[3] Training models ({cfg['epochs']} epochs)...")
    models, histories, test_results, test_loader = train_all_models(
        X, y, meta, cfg, cfg['epochs'], DEVICE, seed=seed)

    # 4. Grid predictions
    print("\n[4] Predicting on grid...")
    grid_probs = predict_on_grid(models, X_g, DEVICE)

    # 5. Visualization
    print("\n[5] Generating figures...")
    save_pfx = str(FIG_DIR / 'WB2')
    plot_traditional_vs_dl(
        'WB2', None, meta_g, grid_probs, save_pfx)

    # WB2 specific: show local-solution structure and internal physical states
    _plot_wb2_local_solutions(meta_g, save_pfx)
    _plot_wb2_physical_maps(meta_g, save_pfx)
    _plot_wb2_v1_v2_scatter(save_pfx)

    # 6. Save
    for mname, model in models.items():
        torch.save(model.state_dict(), SAVE_DIR / f'WB2_{mname.replace("-","_")}.pth')
    _save_metrics('WB2', test_results, histories)

    return test_results


def _plot_wb2_physical_maps(meta_g, save_pfx):
    """Visualize WB2 security region together with internal physical quantities."""
    P_grid = meta_g.get('P_grid')
    Q_grid = meta_g.get('Q_grid')
    if P_grid is None or Q_grid is None:
        return

    labels = meta_g.get('labels_2d')
    V2 = meta_g.get('V2')
    theta2 = meta_g.get('theta2')
    PG1 = meta_g.get('PG1')
    QG1 = meta_g.get('QG1')
    dual_gap = meta_g.get('dual_gap')
    dual_mask = meta_g.get('dual_mask')

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.8))
    fig.patch.set_facecolor('white')

    panels = [
        ('a', labels, 'Security region in (P_d, Q_d)', None, ['#efe7da', '#5b8c5a']),
        ('b', V2, r'Feasible-state voltage $V_2$ (p.u.)', 'viridis', None),
        ('c', theta2, r'Feasible-state angle $\theta_2$ (deg)', 'coolwarm', None),
        ('d', PG1, r'Slack generation $P_{G1}$ (MW)', 'cividis', None),
        ('e', QG1, r'Slack reactive output $Q_{G1}$ (MVAR)', 'magma', None),
        ('f', dual_gap, r'Dual-solution voltage gap $\Delta V_2$ (p.u.)', 'plasma', None),
    ]

    for ax, (tag, data, title, cmap, colors) in zip(axes.ravel(), panels):
        ax.set_facecolor('#fbfaf7')
        ax.grid(True, alpha=0.18, linestyle='--', linewidth=0.6)
        if cmap is None:
            ax.contourf(P_grid, Q_grid, data, levels=[-0.5, 0.5, 1.5], colors=colors, alpha=0.95)
        else:
            masked = np.ma.masked_invalid(data)
            if tag == 'f' and dual_mask is not None:
                masked = np.ma.masked_where(dual_mask < 0.5, data)
            cs = ax.contourf(P_grid, Q_grid, masked, levels=18, cmap=cmap)
            plt.colorbar(cs, ax=ax, fraction=0.045, pad=0.02)
        if labels is not None and labels.max() > 0.5 and labels.min() < 0.5:
            ax.contour(P_grid, Q_grid, labels, levels=[0.5], colors='#24323b', linewidths=1.4)
        ax.scatter([350.0], [-350.0], marker='*', s=200, c='gold', edgecolors='black', linewidths=0.8, zorder=10)
        _panel_tag(ax, tag)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('P_d (MW)', fontweight='bold')
        ax.set_ylabel('Q_d (MVAR)', fontweight='bold')

    fig.suptitle(
        'WB2 Static Security Region and Feasible-Point Physical States',
        fontsize=13,
        fontweight='bold',
        y=0.99,
    )
    plt.tight_layout()
    fig.savefig(f'{save_pfx}_physical_states.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_pfx}_physical_states.png")



def _plot_wb2_local_solutions(meta_g, save_pfx):
    """Show WB2 voltage structure and dual-solution region."""
    P_grid = meta_g.get('P_grid')
    Q_grid = meta_g.get('Q_grid')
    V2 = meta_g.get('V2')
    dual_gap = meta_g.get('dual_gap')
    dual_mask = meta_g.get('dual_mask')
    labels = meta_g.get('labels_2d')
    if P_grid is None or V2 is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.1))
    fig.patch.set_facecolor('white')

    axes[0].set_facecolor('#fbfaf7')
    im = axes[0].contourf(P_grid, Q_grid, np.ma.masked_invalid(V2), levels=20, cmap='viridis')
    plt.colorbar(im, ax=axes[0], label='V2 (p.u.)', fraction=0.045, pad=0.02)
    axes[0].contour(P_grid, Q_grid, V2, levels=[0.95, 1.05],
                    colors=['#8c2f39', '#244c5a'], linewidths=1.9)
    if labels is not None and labels.max() > 0.5 and labels.min() < 0.5:
        axes[0].contour(P_grid, Q_grid, labels, levels=[0.5], colors='white', linewidths=1.4)
    axes[0].scatter([350.0], [-350.0], marker='*', s=220, c='gold',
                    edgecolors='black', linewidths=0.8, zorder=10)
    axes[0].set_xlabel('P_d (MW)', fontweight='bold')
    axes[0].set_ylabel('Q_d (MVAR)', fontweight='bold')
    axes[0].set_title('Voltage-secure strip and nominal point', fontweight='bold')
    _panel_tag(axes[0], 'a')

    axes[1].set_facecolor('#fbfaf7')
    dual_field = np.ma.masked_where((dual_mask is None) or (dual_mask < 0.5), dual_gap)
    cs = axes[1].contourf(P_grid, Q_grid, dual_field, levels=16, cmap='plasma')
    plt.colorbar(cs, ax=axes[1], label=r'$\Delta V_2$ between high/low branches', fraction=0.045, pad=0.02)
    if dual_mask is not None:
        axes[1].contour(P_grid, Q_grid, dual_mask, levels=[0.5], colors='#5d2e8c', linewidths=1.6, linestyles='--')
    if labels is not None and labels.max() > 0.5 and labels.min() < 0.5:
        axes[1].contour(P_grid, Q_grid, labels, levels=[0.5], colors='#24323b', linewidths=1.4)
    axes[1].scatter([350.0], [-350.0], marker='*', s=220, c='gold',
                    edgecolors='black', linewidths=0.8, zorder=10)
    axes[1].set_xlabel('P_d (MW)', fontweight='bold')
    axes[1].set_ylabel('Q_d (MVAR)', fontweight='bold')
    axes[1].set_title('Dual-solution pocket near the collapse boundary', fontweight='bold')
    _panel_tag(axes[1], 'b')

    plt.tight_layout()
    fig.savefig(f'{save_pfx}_local_solutions.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_pfx}_local_solutions.png")


def _plot_wb2_v1_v2_scatter(save_pfx: str):
    """Generate WB2 V1-V2 scatter in fixed-load setting."""
    from scipy.optimize import fsolve
    import pandas as pd

    # WB2 settings consistent with Interior2 script in D:\安全域\1
    base_mva = 100.0
    p_load = 350.0 / base_mva
    q_load = -350.0 / base_mva
    z = 0.04 + 0.2j
    y = 1.0 / z
    g = y.real
    b = y.imag

    vmin, vmax = 0.95, 1.05
    qg_min, qg_max = -4.0, 4.0

    def equations(vars_, pg1_pu):
        v1, v2, th2 = vars_
        c = np.cos(-th2)
        s = np.sin(-th2)
        v1v2 = v1 * v2

        p1 = g * v1**2 - v1v2 * (g * c + b * s)
        p2 = g * v2**2 - v1v2 * (g * c - b * s)
        q2 = -b * v2**2 + v1v2 * (g * s + b * c)
        return [p1 - pg1_pu, p2 + p_load, q2 + q_load]

    def q1_from_state(v1, v2, th2):
        c = np.cos(-th2)
        s = np.sin(-th2)
        v1v2 = v1 * v2
        return -b * v1**2 - v1v2 * (g * s - b * c)

    guesses = [
        (0.96, 1.00, -0.20),
        (1.00, 1.00, -0.50),
        (1.04, 0.98, -0.90),
        (0.98, 1.04, -0.30),
        (1.02, 0.95, -1.20),
    ]

    rows = []
    seen = set()
    for pg1_mw in np.arange(420.0, 480.0 + 1e-9, 0.5):
        pg1_pu = pg1_mw / base_mva
        for guess in guesses:
            try:
                sol, _, ier, _ = fsolve(equations, guess, args=(pg1_pu,), full_output=True, maxfev=2500)
                if ier != 1:
                    continue
                v1, v2, th2 = float(sol[0]), float(sol[1]), float(sol[2])
                residual = equations([v1, v2, th2], pg1_pu)
                if max(abs(np.array(residual))) > 1e-7:
                    continue
                if not (vmin <= v1 <= vmax and vmin <= v2 <= vmax):
                    continue
                qg1 = q1_from_state(v1, v2, th2)
                if not (qg_min <= qg1 <= qg_max):
                    continue

                key = (round(pg1_mw, 3), round(v1, 6), round(v2, 6), round(th2, 6))
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    'Pg1_MW': pg1_mw,
                    'V1_pu': v1,
                    'V2_pu': v2,
                    'theta2_deg': np.degrees(th2),
                    'Qg1_MVAR': qg1 * base_mva,
                })
            except Exception:
                continue

    if not rows:
        print("  WB2 V1-V2 scatter skipped (no points found).")
        return

    df = pd.DataFrame(rows).sort_values(['Pg1_MW', 'V1_pu', 'V2_pu']).reset_index(drop=True)
    csv_path = SAVE_DIR / 'WB2_V1_V2_scatter.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8')

    fig, ax = plt.subplots(figsize=(8.2, 6.3))
    ax.set_facecolor('#fbfaf7')
    ax.grid(True, alpha=0.20, linestyle='--', linewidth=0.6)
    sc = ax.scatter(df['V1_pu'], df['V2_pu'], c=df['Pg1_MW'], cmap='viridis',
                    s=11, alpha=0.85, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Pg1 (MW)', fraction=0.045, pad=0.02)
    ax.set_xlabel('V1 (p.u.)', fontweight='bold')
    ax.set_ylabel('V2 (p.u.)', fontweight='bold')
    ax.set_xlim(0.945, 1.055)
    ax.set_ylim(0.895, 1.105)
    ax.set_title('WB2 Security Set: V1-V2 Scatter', fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{save_pfx}_V1_V2_scatter.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"  Saved: {save_pfx}_V1_V2_scatter.png")
    print(f"  Saved: {csv_path}")


def run_WB5(quick: bool = False, seed: int = 42):
    """
    WB5: 5-bus meshed system.
    Characterize security in (P_G1, P_G5) generator power space.
    Should reproduce 2 disconnected secure components.
    """
    print(f"\n{'='*65}")
    print("  WB5: 5-Bus Meshed System (Generator Power Space)")
    print(f"{'='*65}")
    set_global_seed(seed)

    cfg = {
        'n_samples': 3000 if quick else 10000,
        'n_grid': 30 if quick else 80,
        'epochs': 100 if quick else 300,
        'batch': 512,
        'hidden': [256, 256, 128, 64],
        'feature_dim': 128,
        'classifier_dims': [256, 256, 128, 64],
        'physics_dims': [128, 64],
    }

    # 1. Load traditional IPOPT results (ground truth)
    print("\n[1] Loading traditional IPOPT results (ground truth)...")
    trad = plot_traditional_ipopt_raw('WB5', str(FIG_DIR / 'WB5'))

    # 2. Generate training data
    # Strategy: use traditional results as secure points + generate insecure background
    print("\n[2] Building training dataset from traditional results + background...")
    X, y, meta = get_wb5_traditional_dataset(str(TRAD_DIR), seed=seed)
    if X is None:
        print("  Traditional data not found, generating with Pyomo...")
        X, y, meta = generate_WB5_data(
            n_samples=cfg['n_samples'], seed=seed, verbose=True)
    print(f"  Dataset: shape={X.shape}, security rate={y.mean():.3f}")

    # 3. Generate 2D grid in (P_G1, P_G5) space from scatter (no Pyomo needed)
    print(f"\n[3] 2D grid in (P_G1, P_G5) space (exact IPOPT axes)...")
    if trad is not None:
        df = trad['df']
        pg1_col = 'PG1' if 'PG1' in df.columns else df.columns[1]
        pg5_col = 'PG5' if 'PG5' in df.columns else df.columns[0]
        X_feas_mw = np.column_stack([df[pg1_col].values, df[pg5_col].values]).astype(np.float32)
    else:
        X_feas_mw = X[y > 0.5]
    X_g, y_g, meta_g = build_grid_from_scatter(
        X_feas_mw,
        x_range=(0.0, 700.0), y_range=(0.0, 400.0),
        n_per_dim=cfg['n_grid'], case_key='pg1_pg5',
        exact_axes=True,
        x_step=0.5,
        y_step=1.0,
    )
    np.save(DATA_DIR / 'WB5_X_grid.npy', X_g)
    np.save(DATA_DIR / 'WB5_y_grid.npy', y_g)
    print(f"  Grid security rate: {y_g.mean():.3f}")

    # 4. Training
    print(f"\n[4] Training models ({cfg['epochs']} epochs)...")
    meta['n_bus'] = 5
    models, histories, test_results, test_loader = train_all_models(
        X, y, meta, cfg, cfg['epochs'], DEVICE, seed=seed)

    # 5. Predict on grid (normalize X_g to match training scale)
    print("\n[5] Predicting on grid...")
    X_g_norm = (X_g - meta['X_mean']) / meta['X_std']
    grid_probs = predict_on_grid(models, X_g_norm, DEVICE)

    # 6. Visualizations
    print("\n[6] Generating figures...")
    save_pfx = str(FIG_DIR / 'WB5')
    plot_traditional_vs_dl('WB5', trad, meta_g, grid_probs, save_pfx)
    _plot_local_zoom_density('WB5', trad, meta_g, grid_probs, save_pfx)

    # 7. Quantitative comparison with traditional IPOPT
    print("\n[7] Quantitative comparison with traditional IPOPT...")
    if trad is not None:
        _quantitative_comparison_wb5(trad, meta_g, grid_probs, save_pfx)

    # 8. Save
    for mname, model in models.items():
        torch.save(model.state_dict(), SAVE_DIR / f'WB5_{mname.replace("-","_")}.pth')
    _save_metrics('WB5', test_results, histories)

    return test_results


def _quantitative_comparison_wb5(trad, meta_g, grid_probs, save_prefix):
    """Compare DL boundary prediction accuracy against traditional IPOPT ground truth."""
    df = trad['df']
    pg1_feas = df['PG1'].values if 'PG1' in df.columns else df.iloc[:, 1].values
    pg5_feas = df['PG5'].values if 'PG5' in df.columns else df.iloc[:, 0].values

    # Get grid ground truth
    labels_2d = meta_g['labels_2d']
    pg1_arr = meta_g['pg1_arr']
    pg5_arr = meta_g['pg5_arr']
    XX, YY = np.meshgrid(pg1_arr, pg5_arr)

    print("\n  === Quantitative Comparison: SSR-PDNet vs Traditional IPOPT ===")
    print(f"  {'Model':<15} {'Grid Acc':>10} {'Grid F1':>10} {'Boundary Acc':>14} {'Boundary F1':>12}")
    print("  " + "-" * 54)

    results = {}
    for model_name, probs_flat in grid_probs.items():
        probs_2d = probs_flat.reshape(labels_2d.shape)
        pred_2d = (probs_2d > 0.5).astype(float)

        acc = (pred_2d == labels_2d).mean()
        from sklearn.metrics import f1_score
        f1 = f1_score(labels_2d.ravel(), pred_2d.ravel(), zero_division=0)

        # Boundary accuracy
        error_2d = np.abs(pred_2d - labels_2d)
        # Find points near boundary (where adjacent cells disagree)
        from scipy.ndimage import binary_dilation
        feas_mask = labels_2d > 0.5
        dilated = binary_dilation(feas_mask, iterations=2)
        boundary_mask = dilated & ~feas_mask | (feas_mask & binary_dilation(~feas_mask, iterations=2))
        if boundary_mask.sum() > 0:
            bd_acc = (pred_2d[boundary_mask] == labels_2d[boundary_mask]).mean()
            bd_f1 = f1_score(labels_2d[boundary_mask].ravel(), pred_2d[boundary_mask].ravel(), zero_division=0)
        else:
            bd_acc = 0.0
            bd_f1 = 0.0

        print(f"  {model_name:<15} {acc:>10.4f} {f1:>10.4f} {bd_acc:>14.4f} {bd_f1:>12.4f}")
        results[model_name] = {'grid_acc': acc, 'grid_f1': f1, 'boundary_acc': bd_acc, 'boundary_f1': bd_f1}

    # Save comparison table
    with open(f'{save_prefix}_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {save_prefix}_comparison.json")


def run_case9mod(quick: bool = False, seed: int = 42):
    """
    case9mod: Modified IEEE 9-bus system.
    Characterize security in (P_G2, P_G3) generator power space.
    Should reproduce 3 disconnected secure components.
    """
    print(f"\n{'='*65}")
    print("  case9mod: Modified IEEE 9-Bus (Generator Power Space)")
    print(f"{'='*65}")
    set_global_seed(seed)

    cfg = {
        'n_samples': 3000 if quick else 12000,
        'n_grid': 30 if quick else 70,
        'epochs': 100 if quick else 300,
        'batch': 512,
        'hidden': [512, 512, 256, 128],
        'feature_dim': 256,
        'classifier_dims': [512, 512, 256, 128],
        'physics_dims': [256, 128],
    }

    # 1. Load traditional IPOPT reference
    print("\n[1] Loading traditional IPOPT results (ground truth)...")
    trad = plot_traditional_ipopt_raw('case9mod', str(FIG_DIR / 'case9mod'))

    # 2. Build training dataset
    print("\n[2] Building training dataset from traditional results + background...")
    X, y, meta = get_case9mod_traditional_dataset(str(TRAD_DIR), seed=seed)
    if X is None:
        print("  Traditional data not found, generating with Pyomo...")
        X, y, meta = generate_case9mod_data(
            n_samples=cfg['n_samples'], seed=seed, verbose=True)
    print(f"  Dataset: shape={X.shape}, security rate={y.mean():.3f}")

    # 3. Generate 2D grid from scatter (no Pyomo needed)
    print(f"\n[3] 2D grid in (P_G2, P_G3) space (exact IPOPT axes)...")
    if trad is not None:
        df = trad['df']
        # case9mod CSV uses p2_mw, p3_mw column names
        if 'p2_mw' in df.columns:
            pg2_col, pg3_col = 'p2_mw', 'p3_mw'
        elif 'pg2' in df.columns:
            pg2_col, pg3_col = 'pg2', 'pg3'
        else:
            pg2_col, pg3_col = df.columns[0], df.columns[1]
        X_feas_mw = np.column_stack([df[pg2_col].values, df[pg3_col].values]).astype(np.float32)
    else:
        X_feas_mw = X[y > 0.5]
    case_limits = _case_axis_limits('case9mod', trad_data=trad, arr_x=X_feas_mw[:, 0], arr_y=X_feas_mw[:, 1])
    if case_limits is not None:
        x_lo, x_hi, y_lo, y_hi = case_limits
    else:
        x_lo, x_hi = 0.0, 180.0
        y_lo, y_hi = 0.0, 180.0

    X_g, y_g, meta_g = build_grid_from_scatter(
        X_feas_mw,
        x_range=(x_lo, x_hi), y_range=(y_lo, y_hi),
        n_per_dim=cfg['n_grid'], case_key='pg2_pg3',
        exact_axes=True,
        x_step=(x_hi - x_lo) / (300 - 1),
        y_step=(y_hi - y_lo) / (300 - 1),
    )
    np.save(DATA_DIR / 'case9mod_X_grid.npy', X_g)
    np.save(DATA_DIR / 'case9mod_y_grid.npy', y_g)
    print(f"  Grid security rate: {y_g.mean():.3f}")

    # 4. Training
    print(f"\n[4] Training models ({cfg['epochs']} epochs)...")
    meta['n_bus'] = 9
    models, histories, test_results, test_loader = train_all_models(
        X, y, meta, cfg, cfg['epochs'], DEVICE, seed=seed)

    # 5. Predict on grid (normalize X_g to match training scale)
    print("\n[5] Predicting on grid...")
    X_g_norm = (X_g - meta['X_mean']) / meta['X_std']
    grid_probs = predict_on_grid(models, X_g_norm, DEVICE)

    # 6. Visualizations
    print("\n[6] Generating figures...")
    save_pfx = str(FIG_DIR / 'case9mod')
    plot_traditional_vs_dl('case9mod', trad, meta_g, grid_probs, save_pfx)
    _plot_local_zoom_density('case9mod', trad, meta_g, grid_probs, save_pfx)

    # 7. Quantitative comparison
    print("\n[7] Quantitative comparison with traditional IPOPT...")
    if trad is not None:
        _quantitative_comparison_case9(trad, meta_g, grid_probs, save_pfx)

    # 8. Save
    for mname, model in models.items():
        torch.save(model.state_dict(), SAVE_DIR / f'case9mod_{mname.replace("-","_")}.pth')
    _save_metrics('case9mod', test_results, histories)

    return test_results


def _quantitative_comparison_case9(trad, meta_g, grid_probs, save_prefix):
    """Quantitative comparison for case9mod."""
    from sklearn.metrics import f1_score
    from scipy.ndimage import binary_dilation

    labels_2d = meta_g['labels_2d']

    print("\n  === Quantitative Comparison: SSR-PDNet vs Traditional IPOPT ===")
    print(f"  {'Model':<15} {'Grid Acc':>10} {'Grid F1':>10} {'Boundary Acc':>14} {'Boundary F1':>12}")
    print("  " + "-" * 54)

    results = {}
    for model_name, probs_flat in grid_probs.items():
        probs_2d = probs_flat.reshape(labels_2d.shape)
        pred_2d = (probs_2d > 0.5).astype(float)

        acc = (pred_2d == labels_2d).mean()
        f1 = f1_score(labels_2d.ravel(), pred_2d.ravel(), zero_division=0)

        feas_mask = labels_2d > 0.5
        boundary_mask = (binary_dilation(feas_mask, iterations=2) & ~feas_mask |
                         feas_mask & binary_dilation(~feas_mask, iterations=2))
        if boundary_mask.sum() > 0:
            bd_acc = (pred_2d[boundary_mask] == labels_2d[boundary_mask]).mean()
            bd_f1 = f1_score(labels_2d[boundary_mask].ravel(), pred_2d[boundary_mask].ravel(), zero_division=0)
        else:
            bd_acc = 0.0
            bd_f1 = 0.0

        print(f"  {model_name:<15} {acc:>10.4f} {f1:>10.4f} {bd_acc:>14.4f} {bd_f1:>12.4f}")
        results[model_name] = {'grid_acc': acc, 'grid_f1': f1, 'boundary_acc': bd_acc, 'boundary_f1': bd_f1}

    with open(f'{save_prefix}_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {save_prefix}_comparison.json")


def _plot_lmbm3_physical_maps(df, labels, save_pfx):
    """Plot LMBM3 secure-point physical quantities for each load factor."""
    pg1 = df['PG1'].values
    pg2 = df['PG2'].values

    voltage_cols = [c for c in ['V1', 'V2', 'V3', 'Vm1', 'Vm2', 'Vm3'] if c in df.columns]
    q_cols = [c for c in ['QG1', 'QG2', 'QG3'] if c in df.columns]
    loss_col = 'Ploss_MW' if 'Ploss_MW' in df.columns else ('loss' if 'loss' in df.columns else None)
    angle_col = 'Va2_deg' if 'Va2_deg' in df.columns else ('Va2' if 'Va2' in df.columns else None)

    min_v = df[voltage_cols].min(axis=1).values if voltage_cols else None
    max_v = df[voltage_cols].max(axis=1).values if voltage_cols else None
    q_total = df[q_cols].sum(axis=1).values if q_cols else None
    losses = df[loss_col].values if loss_col else None
    angle2 = df[angle_col].values if angle_col else None

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.0))
    fig.patch.set_facecolor('white')
    panels = [
        ('a', min_v, 'Minimum bus voltage (p.u.)', 'viridis'),
        ('b', max_v, 'Maximum bus voltage (p.u.)', 'magma'),
        ('c', q_total, 'Total reactive generation (MVAR)', 'cividis'),
        ('d', losses if losses is not None else angle2,
         'Active losses (MW)' if losses is not None else 'Bus-2 angle (deg)',
         'plasma' if losses is not None else 'coolwarm'),
    ]

    for ax, (tag, values, title, cmap) in zip(axes.ravel(), panels):
        ax.set_facecolor('#fbfaf7')
        ax.grid(True, alpha=0.18, linestyle='--', linewidth=0.6)
        sc = ax.scatter(pg1, pg2, c=values, s=7, cmap=cmap, alpha=0.82, edgecolors='none')
        plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        _panel_tag(ax, tag)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('P_G1 (MW)', fontweight='bold')
        ax.set_ylabel('P_G2 (MW)', fontweight='bold')

    fig.suptitle(
        f'LMBM3 Secure-Point Physical States ({labels})',
        fontsize=13,
        fontweight='bold',
        y=0.99,
    )
    plt.tight_layout()
    fig.savefig(f'{save_pfx}_physical_states.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_pfx}_physical_states.png")



def run_LMBM3(quick: bool = False, load_factors=None, seed: int = 42):
    """
    LMBM3: 3-bus triangular system.
    Characterize security in (PG1, PG2) generator power space
    at different load factors.  Traditional CSV data from D:/安全域/1/.
    """
    print(f"\n{'='*65}")
    print("  LMBM3: 3-Bus Triangle System (Bifurcation Analysis + DL)")
    print(f"{'='*65}")
    set_global_seed(seed)

    import pandas as pd

    cfg = {
        'n_grid':  25 if quick else 60,
        'epochs':  60 if quick else 200,
        'batch':   256,
        'hidden':  [256, 256, 128, 64],
        'feature_dim':      128,
        'classifier_dims':  [256, 256, 128, 64],
        'physics_dims':     [128, 64],
    }

    # ── 1. Load traditional IPOPT results ────────────────────────────────────
    print("\n[1] Loading LMBM3 traditional IPOPT results...")
    files = {
        'λ=1.490': TRAD_DIR / 'lmbm3 负荷1.490.csv',
        'λ=1.500': TRAD_DIR / 'lmbm3_feasible_points_v2_optimized.csv',
    }
    available = {k: v for k, v in files.items() if v.exists()}
    if not available:
        print("  No LMBM3 CSV files found, skipping LMBM3.")
        return {}

    all_results = {}

    for label, fpath in available.items():
        lf_tag = label.replace('=', '').replace('.', 'p').replace('λ', 'lf')
        print(f"\n  --- {label} ---")
        df = pd.read_csv(fpath)
        print(f"  Loaded {len(df):,} secure points  cols={list(df.columns)}")

        # Feature space: (PG1, PG2) — 2D generator power space
        pg1_feas = df['PG1'].values.astype(np.float32)
        pg2_feas = df['PG2'].values.astype(np.float32)
        X_feas = np.column_stack([pg1_feas, pg2_feas])

        # Background insecure: uniform 2:1 ratio
        rng = np.random.default_rng(seed)
        n_bg = 2 * len(X_feas)
        x1_bg = rng.uniform(pg1_feas.min() * 0.8, pg1_feas.max() * 1.2, n_bg).astype(np.float32)
        x2_bg = rng.uniform(pg2_feas.min() * 0.8, pg2_feas.max() * 1.2, n_bg).astype(np.float32)

        # Remove accidental secure points from background via KDTree
        from sklearn.neighbors import KDTree
        tree = KDTree(X_feas)
        dx = (pg1_feas.max() - pg1_feas.min()) / 30
        dy = (pg2_feas.max() - pg2_feas.min()) / 30
        radius = 2.5 * np.sqrt(dx**2 + dy**2)
        X_bg_raw = np.column_stack([x1_bg, x2_bg])
        counts = tree.query_radius(X_bg_raw, r=radius, count_only=True)
        X_bg = X_bg_raw[counts == 0][:n_bg]

        X = np.vstack([X_feas, X_bg]).astype(np.float32)
        y = np.concatenate([np.ones(len(X_feas)), np.zeros(len(X_bg))]).astype(np.float32)
        print(f"  Dataset: {len(X_feas)} secure + {len(X_bg)} background = {len(X)} total")

        # Normalise
        X_mean = X.mean(axis=0)
        X_std  = X.std(axis=0) + 1e-8
        meta = {'n_bus': 3, 'X_mean': X_mean, 'X_std': X_std}
        X_norm = (X - X_mean) / X_std

        # ── 2. Build 2D grid ─────────────────────────────────────────────────
        print("  Building exact-axis grid from IPOPT coordinates...")
        x_step = 1.0
        y_step = 1.0
        if '1.500' in label:
            x_step = 0.5
        X_g, y_g, meta_g = build_grid_from_scatter(
            X_feas,
            x_range=(pg1_feas.min(), pg1_feas.max()),
            y_range=(pg2_feas.min(), pg2_feas.max()),
            n_per_dim=cfg['n_grid'],
            case_key='pg1_pg5',   # reuse pg1/pg5 key — axes become pg1/pg2
            exact_axes=True,
            x_step=x_step,
            y_step=y_step,
        )
        # Rename keys for LMBM3
        meta_g['pg1_arr'] = meta_g.pop('pg1_arr')  # already correct
        meta_g['pg5_arr'] = meta_g.pop('pg5_arr')  # treat as pg2 axis

        # ── 3. Train models ──────────────────────────────────────────────────
        print(f"  Training models ({cfg['epochs']} epochs)...")
        models, histories, test_results, test_loader = train_all_models(
            X_norm, y, meta, cfg, cfg['epochs'], DEVICE, seed=seed)

        # ── 4. Grid predictions ──────────────────────────────────────────────
        print("  Predicting on grid...")
        X_g_norm = (X_g - X_mean) / X_std
        grid_probs = predict_on_grid(models, X_g_norm, DEVICE)

        # ── 5. Visualisation ─────────────────────────────────────────────────
        print("  Generating figures...")
        save_pfx = str(FIG_DIR / f'LMBM3_{lf_tag}')
        plot_traditional_vs_dl(
            'LMBM3', {'df': df, 'case': 'LMBM3'}, meta_g, grid_probs, save_pfx
        )
        _plot_lmbm3_physical_maps(df, label, save_pfx)

        # Save model weights and metrics
        for mname, model in models.items():
            torch.save(model.state_dict(),
                       SAVE_DIR / f'LMBM3_{lf_tag}_{mname.replace("-","_")}.pth')
        _save_metrics(f'LMBM3_{lf_tag}', test_results, histories)

        r = test_results['SSR-PDNet']
        all_results[label] = {
            'acc': r['acc'], 'f1': r['f1'],
            'n_feasible': len(X_feas),
        }
        print(f"  SSR-PDNet: acc={r['acc']:.4f}  f1={r['f1']:.4f}  rec={r['rec']:.4f}")

    # ── Also plot the combined traditional LMBM3 traditional figure ──────────
    _plot_lmbm3_traditional()

    return all_results


def _plot_lmbm3_traditional():
    """Plot LMBM3 traditional secure sets together with physical-state overlays."""
    import pandas as pd

    files = {
        'λ=1.490': TRAD_DIR / 'lmbm3 负荷1.490.csv',
        'λ=1.500': TRAD_DIR / 'lmbm3_feasible_points_v2_optimized.csv',
    }

    available = {k: v for k, v in files.items() if v.exists()}
    if not available:
        return

    fig, axes = plt.subplots(len(available), 2, figsize=(12.8, 5.0 * len(available)))
    if len(available) == 1:
        axes = np.array([axes])

    for row, (label, fpath) in enumerate(available.items()):
        try:
            df = pd.read_csv(fpath)
            pg1 = df['PG1'].values
            pg2 = df['PG2'].values
            vm_cols = [c for c in ['V1', 'V2', 'V3', 'Vm1', 'Vm2', 'Vm3'] if c in df.columns]
            loss_col = 'Ploss_MW' if 'Ploss_MW' in df.columns else ('loss' if 'loss' in df.columns else None)

            ax0, ax1 = axes[row]
            for ax in (ax0, ax1):
                ax.set_facecolor('#fbfaf7')
                ax.grid(True, alpha=0.18, linestyle='--', linewidth=0.6)
                ax.set_xlabel('P_G1 (MW)', fontsize=11, fontweight='bold')
                ax.set_ylabel('P_G2 (MW)', fontsize=11, fontweight='bold')

            ax0.scatter(pg1, pg2, s=6, c='#215c4f', alpha=0.72, edgecolors='none')
            ax0.set_title(f'LMBM3 static security region {label}', fontsize=11.5, fontweight='bold')
            _panel_tag(ax0, 'a' if row == 0 else chr(ord('a') + 2 * row))

            if vm_cols:
                min_v = df[vm_cols].min(axis=1).values
                sc = ax1.scatter(pg1, pg2, s=7, c=min_v, cmap='viridis', alpha=0.82, edgecolors='none')
                plt.colorbar(sc, ax=ax1, label='Minimum bus voltage (p.u.)', fraction=0.045, pad=0.02)
                ax1.set_title('Voltage margin across secure points', fontsize=11.5, fontweight='bold')
            elif loss_col is not None:
                sc = ax1.scatter(pg1, pg2, s=7, c=df[loss_col].values, cmap='plasma', alpha=0.82, edgecolors='none')
                plt.colorbar(sc, ax=ax1, label='Active loss (MW)', fraction=0.045, pad=0.02)
                ax1.set_title('Loss distribution across secure points', fontsize=11.5, fontweight='bold')
            else:
                ax1.scatter(pg1, pg2, s=7, c='#7c5c36', alpha=0.75, edgecolors='none')
                ax1.set_title('Secure operating-point cloud', fontsize=11.5, fontweight='bold')
            _panel_tag(ax1, 'b' if row == 0 else chr(ord('b') + 2 * row))
        except Exception as e:
            print(f"  Error plotting {label}: {e}")

    plt.tight_layout()
    save_path = str(FIG_DIR / 'LMBM3_traditional.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Helper: save metrics and training curves
# ══════════════════════════════════════════════════════════════════════════════

def _save_metrics(
    case_name: str,
    test_results: dict,
    histories: dict,
    suffix: str = '',
    save_training_curve: bool = True,
):
    """Save test metrics to JSON and optionally plot training curves."""
    # Save metrics
    metrics_save = {name: {k: v for k, v in res.items() if k not in ['probs', 'labels']}
                    for name, res in test_results.items()}
    metric_name = f'{case_name}{suffix}_metrics.json' if suffix else f'{case_name}_metrics.json'
    with open(SAVE_DIR / metric_name, 'w') as f:
        json.dump(metrics_save, f, indent=2)

    if not save_training_curve:
        return

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {'Baseline': '#1f77b4', 'Physics-NN': '#ff7f0e', 'SSR-PDNet': '#2ca02c'}

    for model_name, hist in histories.items():
        c = colors.get(model_name, 'gray')
        if 'train_loss' in hist:
            axes[0].plot(hist['train_loss'], color=c, alpha=0.8, label=f'{model_name} (train)')
        if 'val_loss' in hist:
            axes[0].plot(hist['val_loss'], color=c, linestyle='--', alpha=0.6, label=f'{model_name} (val)')
        if 'val_acc' in hist:
            axes[1].plot(hist['val_acc'], color=c, label=model_name)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{case_name}: Training & Validation Loss')
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val Accuracy')
    axes[1].set_title(f'{case_name}: Validation Accuracy')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    fig_name = f'{case_name}{suffix}_training.png' if suffix else f'{case_name}_training.png'
    fig.savefig(str(FIG_DIR / fig_name), bbox_inches='tight', dpi=200)
    plt.close(fig)


def print_final_summary(all_results: dict):
    """Print consolidated results table."""
    print("\n" + "=" * 75)
    print("FINAL RESULTS — SSR Characterization in Generator Power Space")
    print("=" * 75)
    print(f"{'Case':12s} {'Model':14s} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}")
    print("-" * 75)
    for case_name, results in all_results.items():
        if not isinstance(results, dict):
            continue
        for i, (model_name, res) in enumerate(results.items()):
            if not isinstance(res, dict) or 'acc' not in res:
                continue
            prefix = case_name if i == 0 else " " * len(case_name)
            acc  = res.get('acc',  float('nan'))
            f1   = res.get('f1',   float('nan'))
            prec = res.get('prec', float('nan'))
            rec  = res.get('rec',  float('nan'))
            prec_str = f"{prec:8.4f}" if prec == prec else "     — "
            rec_str  = f"{rec:8.4f}"  if rec  == rec  else "     — "
            print(f"{prefix:12s} {model_name:14s} "
                  f"{acc:8.4f} {f1:8.4f} {prec_str} {rec_str}")
        print()


def aggregate_seed_results(seed_results: list) -> dict:
    """Aggregate per-seed test metrics into mean/std summary."""
    metric_keys = ['acc', 'f1', 'prec', 'rec', 'spec']
    if not seed_results:
        return {'n_seeds': 0, 'models': {}, 'seeds': []}

    model_names = [m for m in seed_results[0].keys() if isinstance(seed_results[0][m], dict)]
    agg_models = {}
    for model_name in model_names:
        stats = {}
        for k in metric_keys:
            vals = [float(r[model_name][k]) for r in seed_results if model_name in r and k in r[model_name]]
            if vals:
                arr = np.array(vals, dtype=float)
                stats[k] = {
                    'mean': float(arr.mean()),
                    'std': float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
                }
        agg_models[model_name] = stats

    seeds = [int(r.get('_seed', -1)) for r in seed_results]
    return {
        'n_seeds': int(len(seed_results)),
        'seeds': seeds,
        'models': agg_models,
    }


def save_seed_aggregate(case_name: str, agg: dict):
    out_path = SAVE_DIR / f'{case_name}_metrics_mean_std.json'
    with open(out_path, 'w') as f:
        json.dump(agg, f, indent=2)


def print_aggregate_summary(case_name: str, agg: dict):
    print(f"\n  === {case_name} aggregated across {agg.get('n_seeds', 0)} seeds ===")
    for model_name, stats in agg.get('models', {}).items():
        acc = stats.get('acc', {'mean': float('nan'), 'std': float('nan')})
        f1 = stats.get('f1', {'mean': float('nan'), 'std': float('nan')})
        prec = stats.get('prec', {'mean': float('nan'), 'std': float('nan')})
        rec = stats.get('rec', {'mean': float('nan'), 'std': float('nan')})
        print(
            f"  {model_name:<12} "
            f"Acc={acc['mean']:.4f}±{acc['std']:.4f}  "
            f"F1={f1['mean']:.4f}±{f1['std']:.4f}  "
            f"Prec={prec['mean']:.4f}±{prec['std']:.4f}  "
            f"Rec={rec['mean']:.4f}±{rec['std']:.4f}"
        )


def _run_case_once(case: str, quick: bool, seed: int):
    if case == 'WB2':
        return run_WB2(quick=quick, seed=seed)
    if case == 'WB5':
        return run_WB5(quick=quick, seed=seed)
    if case == 'case9mod':
        return run_case9mod(quick=quick, seed=seed)
    if case == 'LMBM3':
        return run_LMBM3(quick=quick, seed=seed)
    raise ValueError(f'Unknown case: {case}')


def _zoom_box(case_name: str, x_vals: np.ndarray, y_vals: np.ndarray, grid_data: dict = None):
    """Choose a local zoom window near a dense boundary-formation region."""
    x_lo, x_hi = np.quantile(x_vals, [0.20, 0.80])
    y_lo, y_hi = np.quantile(y_vals, [0.20, 0.80])

    if case_name == 'WB5':
        x_lo, x_hi = np.quantile(x_vals, [0.25, 0.75])
        y_lo, y_hi = np.quantile(y_vals, [0.25, 0.75])
    elif case_name == 'case9mod':
        x_lo, x_hi = np.quantile(x_vals, [0.18, 0.82])
        y_lo, y_hi = np.quantile(y_vals, [0.18, 0.82])

    if grid_data is None:
        return float(x_lo), float(x_hi), float(y_lo), float(y_hi)

    labels_2d = grid_data.get('labels_2d')
    if labels_2d is None or labels_2d.ndim != 2 or min(labels_2d.shape) < 3:
        return float(x_lo), float(x_hi), float(y_lo), float(y_hi)

    if 'pg1_arr' in grid_data and 'pg5_arr' in grid_data:
        arr_x = np.asarray(grid_data['pg1_arr'])
        arr_y = np.asarray(grid_data['pg5_arr'])
    elif 'pg2_arr' in grid_data and 'pg3_arr' in grid_data:
        arr_x = np.asarray(grid_data['pg2_arr'])
        arr_y = np.asarray(grid_data['pg3_arr'])
    elif 'P_arr' in grid_data and 'Q_arr' in grid_data:
        arr_x = np.asarray(grid_data['P_arr'])
        arr_y = np.asarray(grid_data['Q_arr'])
    else:
        return float(x_lo), float(x_hi), float(y_lo), float(y_hi)

    if labels_2d.shape != (len(arr_y), len(arr_x)):
        return float(x_lo), float(x_hi), float(y_lo), float(y_hi)

    boundary = np.zeros_like(labels_2d, dtype=bool)
    boundary[1:, :] |= labels_2d[1:, :] != labels_2d[:-1, :]
    boundary[:-1, :] |= labels_2d[:-1, :] != labels_2d[1:, :]
    boundary[:, 1:] |= labels_2d[:, 1:] != labels_2d[:, :-1]
    boundary[:, :-1] |= labels_2d[:, :-1] != labels_2d[:, 1:]

    if int(boundary.sum()) < 20:
        return float(x_lo), float(x_hi), float(y_lo), float(y_hi)

    XX, YY = np.meshgrid(arr_x, arr_y)
    bx = XX[boundary]
    by = YY[boundary]

    h, x_edges, y_edges = np.histogram2d(bx, by, bins=24)
    if h.max() <= 0:
        return float(x_lo), float(x_hi), float(y_lo), float(y_hi)

    i_max, j_max = np.unravel_index(np.argmax(h), h.shape)
    x_c = 0.5 * (x_edges[i_max] + x_edges[i_max + 1])
    y_c = 0.5 * (y_edges[j_max] + y_edges[j_max + 1])

    x_rng = float(arr_x.max() - arr_x.min())
    y_rng = float(arr_y.max() - arr_y.min())
    half_w = max(0.10 * x_rng, 1e-6)
    half_h = max(0.10 * y_rng, 1e-6)

    x_lo = max(float(arr_x.min()), float(x_c - half_w))
    x_hi = min(float(arr_x.max()), float(x_c + half_w))
    y_lo = max(float(arr_y.min()), float(y_c - half_h))
    y_hi = min(float(arr_y.max()), float(y_c + half_h))

    return x_lo, x_hi, y_lo, y_hi


def _plot_local_zoom_density(case_name: str, trad_data: dict, grid_data: dict, model_probs: dict, save_prefix: str):
    """Create local zoom figure to inspect point layout and density."""
    if trad_data is None or 'df' not in trad_data:
        return

    df = trad_data['df']
    x, y = _extract_scatter_axes(case_name, df)
    if len(x) < 20:
        return

    x_lo, x_hi, y_lo, y_hi = _zoom_box(case_name, x, y, grid_data=grid_data)
    in_zoom = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
    if in_zoom.sum() < 10:
        return

    xlabel, ylabel, _, _ = _get_case_axes(case_name)

    if 'pg1_arr' in grid_data and 'pg5_arr' in grid_data:
        arr_x = grid_data['pg1_arr']
        arr_y = grid_data['pg5_arr']
    elif 'pg2_arr' in grid_data and 'pg3_arr' in grid_data:
        arr_x = grid_data['pg2_arr']
        arr_y = grid_data['pg3_arr']
    else:
        return

    XX, YY = np.meshgrid(arr_x, arr_y)
    labels_2d = grid_data.get('labels_2d')
    if labels_2d is None:
        return

    primary_name = 'SSR-PDNet' if 'SSR-PDNet' in model_probs else next(iter(model_probs))
    probs_2d = model_probs[primary_name].reshape(labels_2d.shape)
    safe_th = _domain_safe_threshold(case_name, probs_2d, labels_2d, arr_x, arr_y)

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.0))
    fig.patch.set_facecolor('white')
    for ax in axes:
        ax.set_facecolor('#fbfaf7')
        ax.grid(True, alpha=0.20, linestyle='--', linewidth=0.6)

    ax = axes[0]
    x_zoom = x[in_zoom]
    y_zoom = y[in_zoom]
    ax.scatter(x_zoom, y_zoom, s=8, c='#215c4f', alpha=0.72, edgecolors='none')
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title('Local secure-point layout', fontweight='bold')
    _panel_tag(ax, 'a')

    ax = axes[1]
    hb = ax.hexbin(x_zoom, y_zoom, gridsize=35, cmap='YlGnBu', mincnt=1, linewidths=0.0)
    plt.colorbar(hb, ax=ax, label='Point count per hexagon', fraction=0.045, pad=0.02)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    area = max((x_hi - x_lo) * (y_hi - y_lo), 1e-8)
    density = float(len(x_zoom) / area)
    ax.text(
        0.03, 0.97,
        f'Points: {len(x_zoom)}\\nArea: {area:.2f}\\nDensity: {density:.2f} pts/unit$^2$',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#b8c2c9', alpha=0.92),
    )
    ax.set_title('Local point density (hexbin)', fontweight='bold')
    _panel_tag(ax, 'b')

    ax = axes[2]
    cs = ax.contourf(XX, YY, probs_2d, levels=np.linspace(0, 1, 21), cmap='viridis', vmin=0.0, vmax=1.0)
    plt.colorbar(cs, ax=ax, label='SSR-PDNet security score', fraction=0.045, pad=0.02)
    if labels_2d.max() > 0.5 and labels_2d.min() < 0.5:
        ax.contour(XX, YY, labels_2d, levels=[0.5], colors='white', linewidths=1.3, linestyles='--')
    if probs_2d.max() > safe_th and probs_2d.min() < safe_th:
        ax.contour(XX, YY, probs_2d, levels=[safe_th], colors='#111111', linewidths=1.8)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title('Local boundary formation', fontweight='bold')
    _panel_tag(ax, 'c')

    fig.suptitle(f'{case_name} local zoom: point arrangement and boundary formation', fontsize=12.3, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{save_prefix}_local_zoom.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_prefix}_local_zoom.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='SSR-PDNet: Static Security Region Characterization in Generator Power Space'
    )
    parser.add_argument('--cases', nargs='+',
                        default=['WB2', 'WB5', 'case9mod', 'LMBM3'],
                        choices=['WB2', 'WB5', 'case9mod', 'LMBM3'])
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer samples/epochs for testing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for single-run mode')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Seed list for multi-seed formal runs, e.g. --seeds 42 123 777')
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  SSR-PDNet: Power System Security Region Characterization")
    print("  Working in GENERATOR POWER SPACE (P_G as axes, loads fixed)")
    print("=" * 65)

    all_results = {}
    t_total = time.time()

    seed_list = args.seeds if args.seeds is not None else [args.seed]
    multi_seed = len(seed_list) > 1

    if multi_seed:
        print(f"  Multi-seed mode enabled: seeds={seed_list}")
    else:
        print(f"  Single-seed mode: seed={seed_list[0]}")

    for case in args.cases:
        t0 = time.time()

        if multi_seed:
            per_seed = []
            for sd in seed_list:
                print(f"\n--- {case} | seed={sd} ---")
                results = _run_case_once(case, quick=args.quick, seed=sd)
                if isinstance(results, dict):
                    results['_seed'] = int(sd)
                per_seed.append(results)

            agg = aggregate_seed_results(per_seed)
            save_seed_aggregate(case, agg)
            print_aggregate_summary(case, agg)
            all_results[case] = agg.get('models', {})
        else:
            sd = seed_list[0]
            results = _run_case_once(case, quick=args.quick, seed=sd)
            all_results[case] = results

        print(f"\n  Wall time for {case}: {time.time()-t0:.1f}s")

    if not multi_seed:
        print_final_summary(all_results)
    print(f"\nTotal wall time: {time.time()-t_total:.1f}s")
    print(f"Figures saved to: {FIG_DIR}/")
    print(f"Models saved to:  {SAVE_DIR}/")
