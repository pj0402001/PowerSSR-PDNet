"""
Data generation specifically for Bukhsh et al. test cases.
Handles the unique characteristics of WB2/WB3 (analytical), WB5, LMBM3, case9mod.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandapower as pp
from scipy.stats.qmc import LatinHypercube
from scipy.optimize import fsolve
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from bukhsh_cases import get_bukhsh_case, create_WB2


# ──────────────────────────────────────────────────────────────
# WB2: Analytical power flow (2-bus, closed form)
# ──────────────────────────────────────────────────────────────

def _wb2_power_flow(P2: float, Q2: float, V1: float = 0.964) -> Tuple[bool, float, float]:
    """
    Solve WB2 power flow analytically.
    Negative Pd/Qd means generation, positive means load.
    Returns (feasible, V2_pu, theta2_deg)
    """
    z = 0.04 + 0.20j
    y = 1.0 / z
    g, b = y.real, y.imag

    def equations(vars):
        V2, th2 = vars
        Pcalc = V2**2 * g - V2 * V1 * (g * np.cos(th2) + b * np.sin(th2))
        Qcalc = -V2**2 * b - V2 * V1 * (g * np.sin(th2) - b * np.cos(th2))
        return [Pcalc - P2, Qcalc - Q2]

    # Try multiple starting points to find the higher-voltage solution
    for v0, th0 in [(1.0, -0.5), (0.8, -1.2), (1.1, -0.3), (0.9, -0.8)]:
        try:
            sol, info, ier, _ = fsolve(equations, [v0, th0], full_output=True)
            if ier == 1:
                V2, th2 = sol
                if V2 > 0 and abs(equations(sol)[0]) < 1e-8:
                    # Check constraints: V in [0.95, 1.05]
                    feasible = (0.95 <= V2 <= 1.05)
                    return feasible, V2, np.degrees(th2)
        except Exception:
            pass
    return False, float('nan'), float('nan')


def generate_WB2_data(
    n_samples: int = 3000,
    load_variation: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    WB2: 2-bus analytical data generation.
    Input space: [P2_load, Q2_load] in MW/MVAR
    Security region defined by: V2 ∈ [0.95, 1.05] and power flow convergence.
    """
    rng = np.random.RandomState(seed)
    # Nominal: P2=350 MW, Q2=-350 MVAR (capacitive)
    P2_nom, Q2_nom = 350.0, -350.0
    baseMVA = 100.0

    # Exploration range
    P_range = np.array([P2_nom * (1 - load_variation), P2_nom * (1 + load_variation)])
    Q_range = np.array([Q2_nom * (1 + load_variation), Q2_nom * (1 - load_variation)])  # reversed for negative

    sampler = LatinHypercube(d=2, seed=seed)
    samples = sampler.random(n=n_samples)
    P2_arr = P_range[0] + samples[:, 0] * (P_range[1] - P_range[0])
    Q2_arr = Q_range[0] + samples[:, 1] * (Q_range[1] - Q_range[0])

    labels = np.zeros(n_samples, dtype=np.float32)
    V2_arr = np.zeros(n_samples)
    th2_arr = np.zeros(n_samples)

    for i in range(n_samples):
        # Power injection (negative = load consuming)
        P_inj = -P2_arr[i] / baseMVA
        Q_inj = -Q2_arr[i] / baseMVA
        feasible, V2, th2 = _wb2_power_flow(P_inj, Q_inj)
        labels[i] = float(feasible)
        V2_arr[i] = V2
        th2_arr[i] = th2

    X = np.column_stack([P2_arr, Q2_arr]).astype(np.float32)
    X_mean = np.array([P2_nom, Q2_nom])
    X_std = np.array([P2_nom * load_variation, abs(Q2_nom) * load_variation])
    X_norm = (X - X_mean) / (X_std + 1e-8)

    meta = {
        'name': 'WB2', 'n_load': 2, 'n_bus': 2,
        'p_base': np.array([P2_nom]),
        'q_base': np.array([Q2_nom]),
        'X_mean': X_mean, 'X_std': X_std,
        'P_raw': P2_arr[:, None], 'Q_raw': Q2_arr[:, None],
        'V2': V2_arr, 'theta2': th2_arr,
        'feasibility_rate': float(labels.mean()),
        'feature_names': ['P2_load (MW)', 'Q2_load (MVAR)'],
        'analytical': True,
    }
    print(f"WB2: {n_samples} samples, feasibility rate = {labels.mean():.3f}")
    return X_norm, labels, meta


def generate_WB2_grid(n_per_dim: int = 60, load_variation: float = 0.5) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Generate 2D grid for WB2 visualization."""
    P2_nom, Q2_nom = 350.0, -350.0
    baseMVA = 100.0

    P_arr = np.linspace(P2_nom * (1 - load_variation), P2_nom * (1 + load_variation), n_per_dim)
    Q_arr = np.linspace(Q2_nom * (1 + load_variation), Q2_nom * (1 - load_variation), n_per_dim)
    PG, QG = np.meshgrid(P_arr, Q_arr)

    P2_flat = PG.ravel()
    Q2_flat = QG.ravel()
    n = len(P2_flat)
    labels = np.zeros(n, dtype=np.float32)
    V2_arr = np.zeros(n)

    for i in range(n):
        P_inj = -P2_flat[i] / baseMVA
        Q_inj = -Q2_flat[i] / baseMVA
        feasible, V2, _ = _wb2_power_flow(P_inj, Q_inj)
        labels[i] = float(feasible)
        V2_arr[i] = V2

    X = np.column_stack([P2_flat, Q2_flat]).astype(np.float32)
    X_mean = np.array([P2_nom, Q2_nom])
    X_std = np.array([P2_nom * load_variation, abs(Q2_nom) * load_variation])
    X_norm = (X - X_mean) / (X_std + 1e-8)

    return X_norm, labels, {
        'P_raw': PG, 'Q_raw': QG, 'V2': V2_arr.reshape(n_per_dim, n_per_dim),
        'n_per_dim': n_per_dim, 'P_arr': P_arr, 'Q_arr': Q_arr,
        'X_mean': X_mean, 'X_std': X_std,
    }


# ──────────────────────────────────────────────────────────────
# Generic pandapower-based data generation (WB5, LMBM3, case9mod)
# ──────────────────────────────────────────────────────────────

def check_operating_point(
    net: pp.pandapowerNet,
    p_load: np.ndarray,
    q_load: np.ndarray,
) -> Tuple[bool, float]:
    """Check if a load configuration lies in the static security region."""
    if len(net.load) > 0:
        net.load['p_mw'] = p_load
        net.load['q_mvar'] = q_load
    try:
        pp.runpp(net, numba=False, verbose=False, tolerance_mva=1e-6, max_iteration=50)
        if not net.converged:
            return False, 1.0

        v = net.res_bus['vm_pu'].values
        v_min_net = net.bus['min_vm_pu'].fillna(0.9).values
        v_max_net = net.bus['max_vm_pu'].fillna(1.1).values
        v_viol = np.maximum(v_min_net - v, 0).sum() + np.maximum(v - v_max_net, 0).sum()

        line_viol = 0.0
        if len(net.line) > 0:
            ll = net.res_line['loading_percent'].values
            line_viol = np.maximum(ll - 100.0, 0).sum() / 100.0

        gen_viol = 0.0
        if len(net.gen) > 0 and len(net.res_gen) > 0:
            pg = net.res_gen['p_mw'].values
            pmin = net.gen['min_p_mw'].fillna(0).values
            pmax = net.gen['max_p_mw'].fillna(1e6).values
            gen_viol = (np.maximum(pmin - pg, 0) + np.maximum(pg - pmax, 0)).sum()
            qg = net.res_gen['q_mvar'].values
            qmin = net.gen['min_q_mvar'].fillna(-1e6).values
            qmax = net.gen['max_q_mvar'].fillna(1e6).values
            gen_viol += (np.maximum(qmin - qg, 0) + np.maximum(qg - qmax, 0)).sum()

        total_viol = v_viol + line_viol + gen_viol
        return total_viol < 1e-3, float(total_viol)
    except Exception:
        return False, 1.0


def generate_bukhsh_data(
    case_name: str,
    n_samples: int = 5000,
    load_variation: float = 0.5,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate labeled SSR dataset for a Bukhsh et al. test case
    using Latin Hypercube Sampling in active power load space.

    Returns normalized feature matrix X, labels y, and metadata dict.
    """
    if case_name == 'WB2':
        return generate_WB2_data(n_samples, load_variation, seed)

    from bukhsh_cases import get_bukhsh_case
    net_base = get_bukhsh_case(case_name)
    p_base = net_base.load['p_mw'].values.copy()
    q_base = net_base.load['q_mvar'].values.copy()
    n_load = len(p_base)

    # Determine load variation range
    # For large load_variation > 1.0, sample in [p_base * min_scale, p_base * max_scale]
    # where min_scale and max_scale are derived from load_variation
    if load_variation <= 1.0:
        # Standard: ±load_variation fraction around base
        p_min = p_base * (1.0 - load_variation)
        p_max = p_base * (1.0 + load_variation)
    else:
        # Wide range: from p_base/load_variation to p_base*load_variation
        # e.g. load_variation=4.9 → [0.1*p_base, 5.0*p_base] centered at 2.55*p_base
        p_min = np.maximum(p_base * 0.1, p_base / (load_variation + 1e-6))
        p_max = p_base * (load_variation + 0.1)

    p_min = np.maximum(p_min, 0.0)  # non-negative loads

    sampler = LatinHypercube(d=n_load, seed=seed)
    raw = sampler.random(n=n_samples)
    P_samples = p_min + raw * (p_max - p_min)

    # Q proportional to P (maintain power factor)
    tan_phi = q_base / (p_base + 1e-8)
    Q_samples = P_samples * tan_phi

    labels = np.zeros(n_samples, dtype=np.float32)
    violations = np.zeros(n_samples, dtype=np.float32)

    if verbose:
        print(f"Generating {n_samples} samples for {case_name}...")

    for i in range(n_samples):
        if verbose and i % 1000 == 0:
            print(f"  {i}/{n_samples} (feas rate so far: {labels[:i].mean():.3f})")
        net = get_bukhsh_case(case_name)
        feasible, viol = check_operating_point(net, P_samples[i], Q_samples[i])
        labels[i] = float(feasible)
        violations[i] = viol

    X = np.hstack([P_samples, Q_samples]).astype(np.float32)
    X_mean = np.concatenate([p_base, q_base])
    X_std = np.maximum(
        np.concatenate([p_max - p_min, np.abs(Q_samples.max(0) - Q_samples.min(0))]),
        1e-6
    ) / 2.0

    X_norm = (X - X_mean) / X_std

    meta = {
        'name': case_name, 'n_load': n_load,
        'n_bus': net_base._metadata['n_bus'],
        'p_base': p_base, 'q_base': q_base,
        'p_min': p_min, 'p_max': p_max,
        'X_mean': X_mean, 'X_std': X_std,
        'P_raw': P_samples, 'Q_raw': Q_samples,
        'violations': violations,
        'feasibility_rate': float(labels.mean()),
        'analytical': False,
    }
    if verbose:
        print(f"Done. Feasibility rate = {labels.mean():.3f}")
    return X_norm, labels, meta


def generate_bukhsh_grid(
    case_name: str,
    n_per_dim: int = 50,
    load_variation: float = 0.5,
    load_idx: Tuple[int, int] = (0, 1),
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate 2D grid in two load dimensions for visualization.
    Other loads are fixed at their base values.
    """
    if case_name == 'WB2':
        return generate_WB2_grid(n_per_dim, load_variation)

    from bukhsh_cases import get_bukhsh_case
    net_base = get_bukhsh_case(case_name)
    p_base = net_base.load['p_mw'].values.copy()
    q_base = net_base.load['q_mvar'].values.copy()
    n_load = len(p_base)
    tan_phi = q_base / (p_base + 1e-8)

    i0, i1 = load_idx
    if load_variation <= 1.0:
        p0_min = p_base[i0] * (1 - load_variation)
        p0_max = p_base[i0] * (1 + load_variation)
        p1_min = p_base[i1] * (1 - load_variation)
        p1_max = p_base[i1] * (1 + load_variation)
    else:
        # Wide range: [0.05*p, load_variation*p] — same formula as generate_bukhsh_data
        p0_min = max(p_base[i0] * 0.05, 0.0)
        p0_max = p_base[i0] * (load_variation + 0.1)
        p1_min = max(p_base[i1] * 0.05, 0.0)
        p1_max = p_base[i1] * (load_variation + 0.1)

    P1_arr = np.linspace(p0_min, p0_max, n_per_dim)
    P2_arr = np.linspace(p1_min, p1_max, n_per_dim)

    G1, G2 = np.meshgrid(P1_arr, P2_arr)
    n = n_per_dim ** 2
    labels = np.zeros(n, dtype=np.float32)

    # Build full P, Q arrays
    P_samples = np.tile(p_base, (n, 1))
    P_samples[:, i0] = G1.ravel()
    P_samples[:, i1] = G2.ravel()
    Q_samples = P_samples * tan_phi

    print(f"Generating {n_per_dim}x{n_per_dim} grid for {case_name}...")
    for i in range(n):
        net = get_bukhsh_case(case_name)
        feasible, _ = check_operating_point(net, P_samples[i], Q_samples[i])
        labels[i] = float(feasible)

    X = np.hstack([P_samples, Q_samples]).astype(np.float32)
    X_mean = np.concatenate([p_base, q_base])
    X_std = np.maximum(np.concatenate([
        p_base * load_variation,
        np.abs(q_base * load_variation) + 1e-6
    ]), 1e-6)
    X_norm = (X - X_mean) / X_std

    meta = {
        'P_grid': G1, 'Q_grid': G2,
        'P1_arr': P1_arr, 'P2_arr': P2_arr,
        'load_idx': (i0, i1),
        'p_base': p_base, 'q_base': q_base,
        'X_mean': X_mean, 'X_std': X_std,
        'n_per_dim': n_per_dim,
        'feasibility_rate': float(labels.mean()),
    }
    print(f"Grid done. Feasibility rate = {labels.mean():.3f}")
    return X_norm, labels, meta
