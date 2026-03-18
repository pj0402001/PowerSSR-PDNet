"""
Power system data generation and feasibility checking.
Supports IEEE 9-bus, 30-bus, 57-bus, and 118-bus test systems.
"""

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pandas as pd
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


def get_test_network(case: str) -> pp.pandapowerNet:
    """Load IEEE test network."""
    loaders = {
        'case9': pn.case9,
        'case30': pn.case30,
        'case57': pn.case57,
        'case118': pn.case118,
    }
    if case not in loaders:
        raise ValueError(f"Unknown case: {case}. Choose from {list(loaders.keys())}")
    return loaders[case]()


def get_network_info(net: pp.pandapowerNet) -> Dict:
    """Extract key network parameters."""
    n_bus = len(net.bus)
    n_gen = len(net.gen) + len(net.ext_grid)
    n_load = len(net.load)
    n_line = len(net.line) + len(net.trafo)

    # Base load
    p_load_base = net.load['p_mw'].values.copy() if len(net.load) > 0 else np.array([])
    q_load_base = net.load['q_mvar'].values.copy() if len(net.load) > 0 else np.array([])

    # Generator limits
    gen_pmin = net.gen['min_p_mw'].values if len(net.gen) > 0 else np.array([])
    gen_pmax = net.gen['max_p_mw'].values if len(net.gen) > 0 else np.array([])

    return {
        'n_bus': n_bus,
        'n_gen': n_gen,
        'n_load': n_load,
        'n_line': n_line,
        'p_load_base': p_load_base,
        'q_load_base': q_load_base,
        'gen_pmin': gen_pmin,
        'gen_pmax': gen_pmax,
    }


def check_feasibility(net: pp.pandapowerNet,
                      p_load: np.ndarray,
                      q_load: np.ndarray,
                      algorithm: str = 'nr') -> Tuple[bool, Dict]:
    """
    Check if given load configuration is feasible (power flow converges
    and all constraints are satisfied).

    Returns:
        feasible: True if the operating point is in the static security region
        info: dictionary with detailed results
    """
    # Set load values
    if len(p_load) > 0:
        net.load['p_mw'] = p_load
        net.load['q_mvar'] = q_load

    try:
        pp.runpp(net, algorithm=algorithm, numba=False, verbose=False,
                 tolerance_mva=1e-6, max_iteration=50)

        if not net.converged:
            return False, {'reason': 'not_converged', 'violations': {}}

        # Check voltage limits
        v_pu = net.res_bus['vm_pu'].values
        v_min = net.bus['min_vm_pu'].fillna(0.9).values
        v_max = net.bus['max_vm_pu'].fillna(1.1).values
        v_viol = np.maximum(v_min - v_pu, 0) + np.maximum(v_pu - v_max, 0)

        # Check line loading
        if len(net.line) > 0:
            loading = net.res_line['loading_percent'].values
            line_viol = np.maximum(loading - 100.0, 0) / 100.0
        else:
            line_viol = np.array([0.0])

        # Check transformer loading
        if len(net.trafo) > 0:
            t_loading = net.res_trafo['loading_percent'].values
            trafo_viol = np.maximum(t_loading - 100.0, 0) / 100.0
        else:
            trafo_viol = np.array([0.0])

        # Check generator limits
        gen_viol = 0.0
        if len(net.gen) > 0:
            p_gen = net.res_gen['p_mw'].values
            p_min = net.gen['min_p_mw'].fillna(-9999).values
            p_max = net.gen['max_p_mw'].fillna(9999).values
            gen_viol = np.sum(np.maximum(p_min - p_gen, 0) + np.maximum(p_gen - p_max, 0))

        total_viol = np.sum(v_viol) + np.sum(line_viol) + np.sum(trafo_viol) + gen_viol

        feasible = (total_viol < 1e-4)

        info = {
            'reason': 'feasible' if feasible else 'constraint_violation',
            'violations': {
                'voltage': float(np.sum(v_viol)),
                'line_loading': float(np.sum(line_viol)),
                'trafo_loading': float(np.sum(trafo_viol)),
                'generator': float(gen_viol),
                'total': float(total_viol),
            },
            'v_pu': v_pu.copy(),
            'loading_percent': loading.copy() if len(net.line) > 0 else np.array([]),
        }
        return feasible, info

    except Exception as e:
        return False, {'reason': f'error: {str(e)}', 'violations': {}}


def generate_security_region_data(
    case: str = 'case9',
    n_samples: int = 5000,
    load_variation: float = 0.5,
    pf_ratio: float = 0.9,
    random_seed: int = 42,
    method: str = 'latin_hypercube',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate labeled dataset for static security region characterization.

    Args:
        case: IEEE test case name
        n_samples: total number of samples to generate
        load_variation: fraction of load that can vary (e.g. 0.5 means ±50%)
        pf_ratio: power factor (cos phi) for reactive power calculation
        random_seed: random seed for reproducibility
        method: sampling method ('random', 'latin_hypercube', 'grid')

    Returns:
        X: (n_samples, 2*n_load) array of [P_load, Q_load] inputs (normalized)
        y: (n_samples,) binary labels (1=feasible, 0=infeasible)
        meta: dict with normalization info and network parameters
    """
    np.random.seed(random_seed)
    net = get_test_network(case)
    info = get_network_info(net)

    p_base = info['p_load_base']
    q_base = info['q_load_base']
    n_load = len(p_base)

    if n_load == 0:
        raise ValueError(f"Network {case} has no loads.")

    # Sampling in load space
    p_min = p_base * (1.0 - load_variation)
    p_max = p_base * (1.0 + load_variation)

    if method == 'latin_hypercube':
        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=n_load, seed=random_seed)
        samples = sampler.random(n=n_samples)
        P_samples = p_min + samples * (p_max - p_min)
    elif method == 'random':
        P_samples = np.random.uniform(p_min, p_max, size=(n_samples, n_load))
    elif method == 'grid':
        # 2D grid for visualization (uses only first 2 loads)
        n_per_dim = int(np.sqrt(n_samples))
        g1 = np.linspace(p_min[0], p_max[0], n_per_dim)
        g2 = np.linspace(p_min[1], p_max[1], n_per_dim) if n_load > 1 else g1
        g1g, g2g = np.meshgrid(g1, g2)
        P_samples = np.zeros((n_per_dim**2, n_load))
        P_samples[:, 0] = g1g.ravel()
        if n_load > 1:
            P_samples[:, 1] = g2g.ravel()
        for i in range(2, n_load):
            P_samples[:, i] = p_base[i]
    else:
        raise ValueError(f"Unknown method: {method}")

    # Q is determined by power factor
    tan_phi = np.sqrt(1 - pf_ratio**2) / pf_ratio
    Q_samples = P_samples * tan_phi

    labels = np.zeros(n_samples, dtype=np.float32)
    violations_list = []

    print(f"Generating {n_samples} samples for {case}...")
    for i in range(n_samples):
        if i % 500 == 0:
            print(f"  Progress: {i}/{n_samples}")

        net_i = get_test_network(case)
        feasible, info_i = check_feasibility(net_i, P_samples[i], Q_samples[i])
        labels[i] = 1.0 if feasible else 0.0
        violations_list.append(info_i.get('violations', {}).get('total', 0.0))

    X = np.hstack([P_samples, Q_samples]).astype(np.float32)

    # Normalization: zero-mean, unit-variance per feature
    X_mean = np.concatenate([p_base, q_base])
    X_std = np.concatenate([
        np.maximum(p_max - p_min, 1e-6) / 2.0,
        np.maximum(p_max - p_min, 1e-6) / 2.0 * tan_phi,
    ])

    X_norm = (X - X_mean) / X_std

    meta = {
        'case': case,
        'n_load': n_load,
        'p_base': p_base,
        'q_base': q_base,
        'p_min': p_min,
        'p_max': p_max,
        'X_mean': X_mean,
        'X_std': X_std,
        'feasibility_rate': float(labels.mean()),
        'violations': np.array(violations_list),
        'P_raw': P_samples,
        'Q_raw': Q_samples,
    }

    print(f"Done. Feasibility rate: {labels.mean():.3f}")
    return X_norm, labels, meta
