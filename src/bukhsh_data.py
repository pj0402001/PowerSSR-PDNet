"""
Data generation for Bukhsh et al. test cases in GENERATOR POWER SPACE.

The Static Security Region (SSR) is defined in
CONTROL VARIABLE SPACE: u = (P_G2, P_G3, ..., P_GN) in MW.
Loads are FIXED parameters at nominal values.

For each candidate (P_G2, ...) point, we solve an AC security-feasibility
problem (objective=0) using Pyomo+IPOPT with multi-start strategy:
  - Flat start: V=1 p.u., theta=0
  - Outer start: pre-specified angles matching MATPOWER initialization

This exactly matches the traditional IPOPT scanning approach in:
  D:/安全域/1/5节点.py (WB5) and D:/安全域/1/case9线路热极限.py (case9mod)

Reference: W. A. Bukhsh et al., "Local Solutions of Optimal Power Flow Problem",
           IEEE Trans. Power Systems, 2013.
"""

import sys
import os
import math
import warnings
import numpy as np
from scipy.stats.qmc import LatinHypercube
from scipy.optimize import fsolve
from typing import Tuple, Dict, Optional, List
warnings.filterwarnings('ignore')

try:
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
    HAS_PYOMO = True
except ImportError:
    HAS_PYOMO = False
    print("WARNING: pyomo not found. Install with: pip install pyomo")

try:
    import pandapower as pp
    HAS_PANDAPOWER = True
except ImportError:
    HAS_PANDAPOWER = False


# ══════════════════════════════════════════════════════════════════
# WB2: Analytical 2-bus system
# Security region: V2 ∈ [0.95, 1.05] (Bukhsh uses 0.95–1.05)
# The control variable is V1 (generator bus voltage setpoint).
# For WB2, we scan (P2_load, Q2_load) since it has 1 gen (slack) and
# the interesting structure is in power injection space.
# But to match the traditional approach: we actually check security feasibility
# as a function of V1 setpoint and load. Since WB2 has only 1 gen,
# the "generator power space" = just P_G1, but P_G1 = P_load + P_loss
# is determined by the load. So for WB2 we scan (P_G1) vs V1 or
# characterize security in (P_load, Q_load) space.
# WB2 security feasibility: can the AC power flow converge with V2 in bounds?
# ══════════════════════════════════════════════════════════════════

def _wb2_solution_details(
    V2: float,
    th2: float,
    V1: float = 0.964,
    baseMVA: float = 100.0,
) -> Dict[str, float]:
    """Return physical quantities for a solved WB2 operating point."""
    z = 0.04 + 0.20j
    y = 1.0 / z

    V1_c = complex(V1, 0.0)
    V2_c = V2 * np.exp(1j * th2)
    I12 = (V1_c - V2_c) * y
    I21 = (V2_c - V1_c) * y
    S12 = V1_c * np.conj(I12)
    S21 = V2_c * np.conj(I21)
    S_loss = S12 + S21

    return {
        'V2': float(V2),
        'theta2': float(np.degrees(th2)),
        'PG1': float(S12.real * baseMVA),
        'QG1': float(S12.imag * baseMVA),
        'P12': float(S12.real * baseMVA),
        'Q12': float(S12.imag * baseMVA),
        'P21': float(S21.real * baseMVA),
        'Q21': float(S21.imag * baseMVA),
        'line_loss': float(S_loss.real * baseMVA),
        'q_loss': float(S_loss.imag * baseMVA),
        'S12': float(abs(S12) * baseMVA),
    }



def _wb2_find_solutions(P2: float, Q2: float, V1: float = 0.964) -> List[Dict[str, float]]:
    """Find distinct WB2 power-flow solutions and return physical-state details."""
    z = 0.04 + 0.20j
    y = 1.0 / z
    g, b = y.real, y.imag

    def equations(vars):
        V2, th2 = vars
        Pcalc = V2**2 * g - V2 * V1 * (g * np.cos(th2) + b * np.sin(th2))
        Qcalc = -V2**2 * b - V2 * V1 * (g * np.sin(th2) - b * np.cos(th2))
        return [Pcalc - P2, Qcalc - Q2]

    guesses = [
        (1.15, -0.15),
        (1.05, -0.10),
        (1.00, -0.30),
        (0.95, -0.80),
        (0.85, -1.10),
        (0.75, -1.35),
    ]
    solutions = []
    for v0, th0 in guesses:
        try:
            sol, _, ier, _ = fsolve(equations, [v0, th0], full_output=True)
            if ier != 1:
                continue
            V2, th2 = float(sol[0]), float(sol[1])
            res = equations(sol)
            if V2 <= 0.1 or abs(res[0]) >= 1e-7 or abs(res[1]) >= 1e-7:
                continue

            details = _wb2_solution_details(V2, th2, V1=V1)
            is_new = all(
                abs(prev['V2'] - details['V2']) > 1e-4
                or abs(prev['theta2'] - details['theta2']) > 1e-2
                for prev in solutions
            )
            if is_new:
                solutions.append(details)
        except Exception:
            pass

    solutions.sort(key=lambda d: d['V2'])
    return solutions



def _wb2_power_flow(P2: float, Q2: float, V1: float = 0.964) -> Tuple[bool, float, float]:
    """
    Solve WB2 power flow: 2-bus, z=0.04+0.20j p.u., V1=0.964 p.u.
    Bus 2 has load (P2, Q2) in p.u. (positive = load consuming).
    Returns (secure, V2_pu, theta2_deg) where security requires V2 in [0.95,1.05].
    """
    solutions = _wb2_find_solutions(P2, Q2, V1=V1)
    if solutions:
        best = solutions[-1]
        feasible = (0.95 <= best['V2'] <= 1.05)
        return feasible, best['V2'], best['theta2']
    return False, float('nan'), float('nan')


def generate_WB2_data(
    n_samples: int = 5000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    WB2: Scan (P_G1, V1) space.
    Since WB2 has single slack generator, we characterize security
    in (P_load_2, Q_load_2) space (equivalent to generator output space
    as P_G1 = P_load + P_loss, Q_G1 = Q_load + Q_loss).

    The security-region boundary in (P_load, Q_load) space shows the
    multiple local solutions structure from Bukhsh et al. (2013).
    """
    # Nominal: P2=350 MW, Q2=-350 MVAR (capacitive load)
    P2_nom, Q2_nom = 350.0, -350.0
    baseMVA = 100.0

    # Scan range matching Bukhsh et al. test case
    # P2_load ∈ [100, 600] MW, Q2_load ∈ [-700, 0] MVAR
    P2_min, P2_max = 100.0, 600.0
    Q2_min, Q2_max = -700.0, 0.0

    sampler = LatinHypercube(d=2, seed=seed)
    samples = sampler.random(n=n_samples)
    P2_arr = P2_min + samples[:, 0] * (P2_max - P2_min)
    Q2_arr = Q2_min + samples[:, 1] * (Q2_max - Q2_min)

    labels = np.zeros(n_samples, dtype=np.float32)
    V2_arr = np.full(n_samples, np.nan)
    th2_arr = np.full(n_samples, np.nan)
    PG1_arr = np.full(n_samples, np.nan)
    QG1_arr = np.full(n_samples, np.nan)
    loss_arr = np.full(n_samples, np.nan)
    S12_arr = np.full(n_samples, np.nan)
    dual_gap_arr = np.full(n_samples, np.nan)
    has_dual_arr = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        P_inj = -P2_arr[i] / baseMVA
        Q_inj = -Q2_arr[i] / baseMVA
        sols = _wb2_find_solutions(P_inj, Q_inj)
        if sols:
            high = sols[-1]
            low = sols[0]
            feasible = float(0.95 <= high['V2'] <= 1.05)
            labels[i] = feasible
            V2_arr[i] = high['V2']
            th2_arr[i] = high['theta2']
            PG1_arr[i] = high['PG1']
            QG1_arr[i] = high['QG1']
            loss_arr[i] = high['line_loss']
            S12_arr[i] = high['S12']
            if len(sols) >= 2:
                dual_gap_arr[i] = high['V2'] - low['V2']
                has_dual_arr[i] = 1.0

    X = np.column_stack([P2_arr, Q2_arr]).astype(np.float32)
    X_mean = np.array([P2_nom, Q2_nom], dtype=np.float32)
    X_std = np.array([250.0, 350.0], dtype=np.float32)
    X_norm = (X - X_mean) / X_std

    meta = {
        'name': 'WB2',
        'case': 'WB2',
        'n_gen': 1,
        'n_bus': 2,
        'control_vars': ['P2_load (MW)', 'Q2_load (MVAR)'],
        'feature_names': ['P2_load (MW)', 'Q2_load (MVAR)'],
        'X_mean': X_mean,
        'X_std': X_std,
        'P_raw': P2_arr,
        'Q_raw': Q2_arr,
        'V2': V2_arr,
        'theta2': th2_arr,
        'PG1': PG1_arr,
        'QG1': QG1_arr,
        'line_loss': loss_arr,
        'line_loading': S12_arr,
        'dual_gap': dual_gap_arr,
        'has_dual': has_dual_arr,
        'security_rate': float(labels.mean()),
        'analytical': True,
        'space': 'load_space_2bus',
    }
    print(f"WB2: {n_samples} samples, security rate = {labels.mean():.3f}")
    return X_norm, labels, meta


def generate_WB2_grid(
    n_per_dim: int = 80,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Generate 2D grid for WB2 visualization in (P2_load, Q2_load) space."""
    P2_arr = np.linspace(100.0, 600.0, n_per_dim)
    Q2_arr = np.linspace(-700.0, 0.0, n_per_dim)
    PG, QG = np.meshgrid(P2_arr, Q2_arr)

    P2_flat = PG.ravel()
    Q2_flat = QG.ravel()
    n = len(P2_flat)
    labels = np.zeros(n, dtype=np.float32)
    V2_grid = np.full(n, np.nan)
    th2_grid = np.full(n, np.nan)
    PG1_grid = np.full(n, np.nan)
    QG1_grid = np.full(n, np.nan)
    loss_grid = np.full(n, np.nan)
    loading_grid = np.full(n, np.nan)
    dual_gap_grid = np.full(n, np.nan)
    dual_mask = np.zeros(n, dtype=np.float32)
    baseMVA = 100.0

    for i in range(n):
        P_inj = -P2_flat[i] / baseMVA
        Q_inj = -Q2_flat[i] / baseMVA
        sols = _wb2_find_solutions(P_inj, Q_inj)
        if not sols:
            continue

        high = sols[-1]
        low = sols[0]
        labels[i] = float(0.95 <= high['V2'] <= 1.05)
        V2_grid[i] = high['V2']
        th2_grid[i] = high['theta2']
        PG1_grid[i] = high['PG1']
        QG1_grid[i] = high['QG1']
        loss_grid[i] = high['line_loss']
        loading_grid[i] = high['S12']
        if len(sols) >= 2:
            dual_gap_grid[i] = high['V2'] - low['V2']
            dual_mask[i] = 1.0

    P2_nom, Q2_nom = 350.0, -350.0
    X = np.column_stack([P2_flat, Q2_flat]).astype(np.float32)
    X_mean = np.array([P2_nom, Q2_nom], dtype=np.float32)
    X_std = np.array([250.0, 350.0], dtype=np.float32)
    X_norm = (X - X_mean) / X_std

    return X_norm, labels, {
        'P_grid': PG,
        'Q_grid': QG,
        'P_arr': P2_arr,
        'Q_arr': Q2_arr,
        'V2': V2_grid.reshape(n_per_dim, n_per_dim),
        'theta2': th2_grid.reshape(n_per_dim, n_per_dim),
        'PG1': PG1_grid.reshape(n_per_dim, n_per_dim),
        'QG1': QG1_grid.reshape(n_per_dim, n_per_dim),
        'line_loss': loss_grid.reshape(n_per_dim, n_per_dim),
        'line_loading': loading_grid.reshape(n_per_dim, n_per_dim),
        'dual_gap': dual_gap_grid.reshape(n_per_dim, n_per_dim),
        'dual_mask': dual_mask.reshape(n_per_dim, n_per_dim),
        'labels_2d': labels.reshape(n_per_dim, n_per_dim),
        'X_mean': X_mean,
        'X_std': X_std,
        'axis_labels': ('P2_load (MW)', 'Q2_load (MVAR)'),
    }


# ══════════════════════════════════════════════════════════════════
# Pyomo/IPOPT-based feasibility checker for multi-generator systems
# ══════════════════════════════════════════════════════════════════

def _build_ybus(bus_data, branch_data):
    """Build Y-bus admittance matrix from bus/branch data (1-indexed)."""
    n_bus = bus_data.shape[0]
    Ybus = np.zeros((n_bus, n_bus), dtype=complex)
    for br in branch_data:
        i = int(br[0]) - 1
        j = int(br[1]) - 1
        r, x, b_sh = br[2], br[3], br[4]
        z = complex(r, x)
        y = 1.0 / z if abs(z) > 1e-10 else complex(0, 1.0/x)
        bsh = complex(0, b_sh / 2.0)
        Ybus[i, i] += y + bsh
        Ybus[j, j] += y + bsh
        Ybus[i, j] -= y
        Ybus[j, i] -= y
    return Ybus.real, Ybus.imag  # G, B matrices


class PyomoFeasibilityChecker:
    """
    AC power flow feasibility checker using Pyomo + IPOPT.
    Matches exactly the traditional method from D:/安全域/1/5节点.py

    For each candidate generator output point (P_G2, P_G3, ...),
    fixes those values and solves the AC power flow feasibility problem
    (objective = constant 0) with multi-start initialization.
    """

    def __init__(self, system_config: Dict):
        """
        system_config keys:
          - bus: np.array shape (n_bus, 13) in MATPOWER format
          - gen: np.array shape (n_gen, >=10) in MATPOWER format
          - branch: np.array shape (n_br, >=13) in MATPOWER format
          - baseMVA: float
          - slack_bus: int (1-indexed)
          - fixed_gen_indices: list of 1-indexed gen indices to fix (e.g. [2] or [2,3])
          - outer_Va_deg: dict {bus_idx: angle_deg} for outer start
          - solver_tol: float (default 1e-8)
          - solver_max_iter: int (default 3000)
        """
        self.cfg = system_config
        self.baseMVA = system_config.get('baseMVA', 100.0)
        self.slack_bus = system_config.get('slack_bus', 1)

        bus = system_config['bus']
        gen = system_config['gen']
        branch = system_config['branch']

        self.n_bus = bus.shape[0]
        self.n_gen = gen.shape[0]
        self.n_branch = branch.shape[0]

        # Bus data
        self.buses = list(range(1, self.n_bus + 1))
        self.Pd = {int(bus[i, 0]): bus[i, 2] / self.baseMVA for i in range(self.n_bus)}
        self.Qd = {int(bus[i, 0]): bus[i, 3] / self.baseMVA for i in range(self.n_bus)}
        self.Vmax = {int(bus[i, 0]): bus[i, 11] for i in range(self.n_bus)}
        self.Vmin = {int(bus[i, 0]): bus[i, 12] for i in range(self.n_bus)}

        # Generator data
        self.gens = list(range(1, self.n_gen + 1))
        self.gen_bus = {i + 1: int(gen[i, 0]) for i in range(self.n_gen)}
        self.Pmax = {i + 1: gen[i, 8] / self.baseMVA for i in range(self.n_gen)}
        self.Pmin = {i + 1: gen[i, 9] / self.baseMVA for i in range(self.n_gen)}
        self.Qmax = {i + 1: gen[i, 3] / self.baseMVA for i in range(self.n_gen)}
        self.Qmin = {i + 1: gen[i, 4] / self.baseMVA for i in range(self.n_gen)}

        # Branch data
        self.lines = list(range(1, self.n_branch + 1))
        self.fbus = {l + 1: int(branch[l, 0]) for l in range(self.n_branch)}
        self.tbus = {l + 1: int(branch[l, 1]) for l in range(self.n_branch)}
        self.rateA = {l + 1: branch[l, 5] / self.baseMVA for l in range(self.n_branch)}

        # Compute G, B matrices
        G, B = _build_ybus(bus, branch)
        self.G = G
        self.B = B

        # Series admittances for line flow constraints
        self.g_series = {}
        self.b_series = {}
        self.b_charge = {}
        for l in range(self.n_branch):
            r, x, bsh = branch[l, 2], branch[l, 3], branch[l, 4]
            z = complex(r, x)
            if abs(z) > 1e-10:
                y = 1.0 / z
            else:
                y = complex(0, 1.0 / x)
            self.g_series[l + 1] = y.real
            self.b_series[l + 1] = y.imag
            self.b_charge[l + 1] = bsh

        self.fixed_gens = system_config.get('fixed_gen_indices', [])
        self.outer_Va = system_config.get('outer_Va_deg', {})
        self.solver_tol = system_config.get('solver_tol', 1e-8)
        self.solver_max_iter = system_config.get('solver_max_iter', 3000)

    def _build_model(self, pg_fixed_pu: Dict[int, float],
                     init_type: str = 'flat',
                     warm: Optional[Dict] = None) -> 'pyo.ConcreteModel':
        """Build Pyomo feasibility model with given fixed P_G values."""
        m = pyo.ConcreteModel()
        m.BUS = pyo.Set(initialize=self.buses)
        m.GEN = pyo.Set(initialize=self.gens)
        m.LINE = pyo.Set(initialize=self.lines)

        # Variables
        m.Vm = pyo.Var(m.BUS, within=pyo.Reals, initialize=1.0)
        m.Va = pyo.Var(m.BUS, within=pyo.Reals, initialize=0.0)
        m.Pg = pyo.Var(m.GEN, within=pyo.Reals, initialize=0.0)
        m.Qg = pyo.Var(m.GEN, within=pyo.Reals, initialize=0.0)

        # Bounds
        for i in m.BUS:
            m.Vm[i].setlb(self.Vmin[i])
            m.Vm[i].setub(self.Vmax[i])
        for g in m.GEN:
            m.Pg[g].setlb(self.Pmin[g])
            m.Pg[g].setub(self.Pmax[g])
            m.Qg[g].setlb(self.Qmin[g])
            m.Qg[g].setub(self.Qmax[g])

        # Slack bus angle reference
        slack = self.slack_bus
        m.ref_angle = pyo.Constraint(expr=m.Va[slack] == 0.0)

        # Fix specified generator active powers
        for g, pg_pu in pg_fixed_pu.items():
            m.Pg[g].fix(pg_pu)

        # Constant objective (pure feasibility)
        m.OBJ = pyo.Objective(expr=0.0, sense=pyo.minimize)

        # AC power flow balance constraints
        G, B = self.G, self.B

        def P_balance(m, i):
            Pi = sum(
                m.Vm[i] * m.Vm[j] * (
                    G[i-1, j-1] * pyo.cos(m.Va[i] - m.Va[j]) +
                    B[i-1, j-1] * pyo.sin(m.Va[i] - m.Va[j])
                )
                for j in m.BUS
            )
            Pg_sum = sum(m.Pg[g] for g in m.GEN if self.gen_bus[g] == i)
            return Pi == Pg_sum - self.Pd[i]

        def Q_balance(m, i):
            Qi = sum(
                m.Vm[i] * m.Vm[j] * (
                    G[i-1, j-1] * pyo.sin(m.Va[i] - m.Va[j]) -
                    B[i-1, j-1] * pyo.cos(m.Va[i] - m.Va[j])
                )
                for j in m.BUS
            )
            Qg_sum = sum(m.Qg[g] for g in m.GEN if self.gen_bus[g] == i)
            return Qi == Qg_sum - self.Qd[i]

        m.P_balance = pyo.Constraint(m.BUS, rule=P_balance)
        m.Q_balance = pyo.Constraint(m.BUS, rule=Q_balance)

        # Apparent power line flow limits |S_ft|^2 <= rateA^2
        m.Sf_limit = pyo.ConstraintList()
        m.St_limit = pyo.ConstraintList()
        for l in m.LINE:
            i, j = self.fbus[l], self.tbus[l]
            g_s = self.g_series[l]
            b_s = self.b_series[l]
            bc = self.b_charge[l]
            ra = self.rateA[l]
            if ra > 999.0:  # effectively unconstrained
                continue

            def Pft(m, i=i, j=j, gs=g_s, bs=b_s):
                return (gs * m.Vm[i]**2
                        - m.Vm[i] * m.Vm[j] * (gs * pyo.cos(m.Va[i] - m.Va[j])
                                                 + bs * pyo.sin(m.Va[i] - m.Va[j])))

            def Qft(m, i=i, j=j, gs=g_s, bs=b_s, bc=bc):
                return (-(bs + bc / 2.0) * m.Vm[i]**2
                        + m.Vm[i] * m.Vm[j] * (bs * pyo.cos(m.Va[i] - m.Va[j])
                                                 - gs * pyo.sin(m.Va[i] - m.Va[j])))

            def Ptf(m, i=i, j=j, gs=g_s, bs=b_s):
                return (gs * m.Vm[j]**2
                        - m.Vm[i] * m.Vm[j] * (gs * pyo.cos(m.Va[i] - m.Va[j])
                                                 - bs * pyo.sin(m.Va[i] - m.Va[j])))

            def Qtf(m, i=i, j=j, gs=g_s, bs=b_s, bc=bc):
                return (-(bs + bc / 2.0) * m.Vm[j]**2
                        + m.Vm[i] * m.Vm[j] * (bs * pyo.cos(m.Va[i] - m.Va[j])
                                                 + gs * pyo.sin(m.Va[i] - m.Va[j])))

            m.Sf_limit.add(expr=Pft(m)**2 + Qft(m)**2 <= ra**2)
            m.St_limit.add(expr=Ptf(m)**2 + Qtf(m)**2 <= ra**2)

        # Initialize
        if init_type == 'outer' and self.outer_Va:
            for i in m.BUS:
                m.Va[i].value = math.radians(self.outer_Va.get(i, 0.0))
                m.Vm[i].value = 1.0
        else:
            for i in m.BUS:
                m.Vm[i].value = 1.0
                m.Va[i].value = 0.0

        # Warm start override
        if warm is not None:
            for i in m.BUS:
                if 'Vm' in warm and i in warm['Vm']:
                    m.Vm[i].value = warm['Vm'][i]
                if 'Va' in warm and i in warm['Va']:
                    m.Va[i].value = warm['Va'][i]
            for g in m.GEN:
                if 'Qg' in warm and g in warm['Qg']:
                    m.Qg[g].value = warm['Qg'][g]

        return m

    def try_solve(self, pg_fixed_pu: Dict[int, float],
                  init_type: str = 'flat',
                  warm: Optional[Dict] = None) -> Tuple[bool, Optional[Dict]]:
        """Try to solve feasibility with given initialization. Returns (feasible, warm_state)."""
        import io, contextlib
        try:
            m = self._build_model(pg_fixed_pu, init_type, warm)
            solver = pyo.SolverFactory('ipopt')
            solver.options['tol'] = self.solver_tol
            solver.options['acceptable_tol'] = self.solver_tol * 100
            solver.options['max_iter'] = self.solver_max_iter
            solver.options['print_level'] = 0
            solver.options['sb'] = 'yes'  # suppress banner

            # Suppress IPOPT output
            with open(os.devnull, 'w') as devnull:
                old_stdout, old_stderr = sys.stdout, sys.stderr
                try:
                    sys.stdout = devnull
                    sys.stderr = devnull
                    res = solver.solve(m, tee=False)
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

            term = res.solver.termination_condition
            stat = res.solver.status

            feasible = (
                stat in (pyo.SolverStatus.ok, pyo.SolverStatus.warning) and
                term in (pyo.TerminationCondition.optimal,
                         pyo.TerminationCondition.locallyOptimal,
                         pyo.TerminationCondition.feasible)
            )

            warm_state = None
            if feasible:
                warm_state = {
                    'Vm': {i: pyo.value(m.Vm[i]) for i in m.BUS},
                    'Va': {i: pyo.value(m.Va[i]) for i in m.BUS},
                    'Qg': {g: pyo.value(m.Qg[g]) for g in m.GEN},
                }

            return feasible, warm_state

        except Exception:
            return False, None

    def check_multistart(self, pg_fixed_pu: Dict[int, float],
                         warm_pools: Optional[Dict] = None,
                         direction: str = 'inc') -> Tuple[bool, Optional[Dict], Dict]:
        """
        Multi-start feasibility check: flat start + outer start, with warm-starting.
        Returns (feasible, warm_state_if_feasible, updated_warm_pools).
        """
        if warm_pools is None:
            warm_pools = {
                ('inc', 'flat'): None, ('inc', 'outer'): None,
                ('dec', 'flat'): None, ('dec', 'outer'): None,
            }

        for init_type in ['flat', 'outer']:
            key = (direction, init_type)
            warm = warm_pools.get(key)
            feasible, warm_new = self.try_solve(pg_fixed_pu, init_type, warm)
            if feasible:
                warm_pools[key] = warm_new
                return True, warm_new, warm_pools

        # All starts failed: clear pools to avoid bad warm-start propagation
        warm_pools[(direction, 'flat')] = None
        warm_pools[(direction, 'outer')] = None
        return False, None, warm_pools


# ══════════════════════════════════════════════════════════════════
# WB5 System Configuration
# ══════════════════════════════════════════════════════════════════

def _get_wb5_config() -> Dict:
    """WB5 system data in MATPOWER format (matching D:/安全域/1/5节点.py exactly)."""
    baseMVA = 100.0
    # bus: [bus_id, type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin]
    bus = np.array([
        [1, 3,   0,   0, 0, 0, 1, 1.0,    0.0, 345, 1, 1.13, 0.87],
        [2, 1, 130,  20, 0, 0, 1, 1.0,  -10.0, 345, 1, 1.13, 0.87],
        [3, 1, 130,  20, 0, 0, 1, 1.0,  -20.0, 345, 1, 1.13, 0.87],
        [4, 1,  65,  10, 0, 0, 1, 1.0, -135.0, 345, 1, 1.13, 0.87],
        [5, 2,   0,   0, 0, 0, 1, 1.0, -140.0, 345, 1, 1.13, 0.87],
    ], dtype=float)

    # gen: [bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin]
    gen = np.array([
        [1, 500,  0, 1800, -30, 1.0, 100, 1, 5000,   0],  # G1 at bus 1 (slack)
        [5,   0,  0, 1800, -30, 1.0, 100, 1, 5000,   0],  # G5 at bus 5
    ], dtype=float)

    # branch: [from, to, r, x, b, rateA, rateB, rateC, tap, shift, status, angmin, angmax]
    branch = np.array([
        [1, 2, 0.04, 0.09, 0.00, 2500, 2500, 2500, 0, 0, 1, -360, 360],
        [1, 3, 0.05, 0.10, 0.00, 2500, 2500, 2500, 0, 0, 1, -360, 360],
        [2, 4, 0.55, 0.90, 0.45, 2500, 2500, 2500, 0, 0, 1, -360, 360],
        [3, 5, 0.55, 0.90, 0.45, 2500, 2500, 2500, 0, 0, 1, -360, 360],
        [4, 5, 0.06, 0.10, 0.00, 2500, 2500, 2500, 0, 0, 1, -360, 360],
        [2, 3, 0.07, 0.09, 0.00, 2500, 2500, 2500, 0, 0, 1, -360, 360],
    ], dtype=float)

    return {
        'bus': bus, 'gen': gen, 'branch': branch,
        'baseMVA': baseMVA,
        'slack_bus': 1,
        'fixed_gen_indices': [2],   # Fix G5 (gen index 2 = gen at bus 5)
        # Outer Va from MATPOWER solution angles
        'outer_Va_deg': {1: 0.0, 2: -10.0, 3: -20.0, 4: -135.0, 5: -140.0},
        'solver_tol': 1e-8,
        'solver_max_iter': 3000,
    }


def generate_WB5_data(
    n_samples: int = 5000,
    seed: int = 42,
    pg5_range: Tuple[float, float] = (0.0, 400.0),
    use_lhs: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    WB5: Generate SSR dataset in (P_G5) space.
    Loads fixed at nominal: [130, 130, 65] MW at buses [2, 3, 4].

    Control variable: P_G5 (generator at bus 5) in [0, 400] MW.
    P_G1 (slack bus) is determined by power balance.

    Input feature: [P_G5] in MW (1D)
    For visualization: 1D sweep P_G5 ∈ [0, 400] MW

    Note: WB5 has 2 disconnected feasible components in P_G5 space.
    The boundary is characterized by the presence of local solutions.
    """
    if not HAS_PYOMO:
        raise RuntimeError("Pyomo required. Install: pip install pyomo")

    cfg = _get_wb5_config()
    checker = PyomoFeasibilityChecker(cfg)
    baseMVA = cfg['baseMVA']

    pg5_min, pg5_max = pg5_range

    if use_lhs:
        sampler = LatinHypercube(d=1, seed=seed)
        samples = sampler.random(n=n_samples)
        pg5_arr = pg5_min + samples[:, 0] * (pg5_max - pg5_min)
    else:
        pg5_arr = np.linspace(pg5_min, pg5_max, n_samples)

    labels = np.zeros(n_samples, dtype=np.float32)

    if verbose:
        print(f"WB5: Generating {n_samples} samples in P_G5 ∈ [{pg5_min:.0f}, {pg5_max:.0f}] MW...")

    warm_pools = {
        ('inc', 'flat'): None, ('inc', 'outer'): None,
        ('dec', 'flat'): None, ('dec', 'outer'): None,
    }

    for i, pg5_MW in enumerate(pg5_arr):
        if verbose and i % 500 == 0:
            print(f"  {i}/{n_samples}, feas rate: {labels[:i].mean():.3f}")
        pg_fixed = {2: pg5_MW / baseMVA}
        feasible, _, warm_pools = checker.check_multistart(pg_fixed, warm_pools, 'inc')
        labels[i] = float(feasible)

    X = pg5_arr[:, None].astype(np.float32)
    pg5_nom = 200.0
    X_mean = np.array([pg5_nom], dtype=np.float32)
    X_std = np.array([pg5_max - pg5_min], dtype=np.float32) / 2.0
    X_norm = (X - X_mean) / X_std

    if verbose:
        print(f"WB5: security rate = {labels.mean():.3f}")

    return X_norm, labels, {
        'name': 'WB5',
        'case': 'WB5',
        'n_gen': 2,
        'n_bus': 5,
        'control_vars': ['P_G5 (MW)'],
        'feature_names': ['P_G5 (MW)'],
        'pg5_raw': pg5_arr,
        'pg5_range': pg5_range,
        'X_mean': X_mean,
        'X_std': X_std,
        'security_rate': float(labels.mean()),
        'space': 'generator_power_space',
    }


def generate_WB5_grid(
    n_per_dim: int = 200,
    pg1_range: Tuple[float, float] = (0.0, 700.0),
    pg5_range: Tuple[float, float] = (0.0, 400.0),
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    WB5: 2D grid in (P_G1 conceptual, P_G5) space.

    Since P_G1 is the slack bus (determined by power balance), we scan P_G5
    and for each P_G5 value, attempt multiple solver starts. The 2D "region"
    is actually swept line by line: for each P_G5 row, we scan whether the
    system is feasible (P_G1 gets determined automatically).

    For true 2D visualization matching D:/安全域/1/5节点数据.csv:
    We sweep P_G5 ∈ [0,400] (Y-axis) and for each P_G5, record the
    feasible P_G1 range (X-axis) found by IPOPT.

    Returns 2D boolean grid in (P_G1, P_G5) space.
    """
    if not HAS_PYOMO:
        raise RuntimeError("Pyomo required. Install: pip install pyomo")

    cfg = _get_wb5_config()
    checker = PyomoFeasibilityChecker(cfg)
    baseMVA = cfg['baseMVA']

    pg5_arr = np.linspace(pg5_range[0], pg5_range[1], n_per_dim)
    pg1_arr = np.linspace(pg1_range[0], pg1_range[1], n_per_dim)

    # Grid: rows = P_G5, cols = P_G1
    # For each (P_G5, P_G1): P_G1 is the slack, so we CANNOT fix both.
    # Instead, we use the traditional approach:
    # Fix P_G5, let P_G1 float as slack. Check if solution exists.
    # The "P_G1" in the results is what IPOPT found as the slack output.
    #
    # To get a 2D grid like the traditional results, we track the actual P_G1
    # that IPOPT finds for each feasible P_G5 point.

    feasible_pg5 = []
    feasible_pg1 = []

    if verbose:
        print(f"WB5: Scanning {n_per_dim} P_G5 values in [{pg5_range[0]:.0f}, {pg5_range[1]:.0f}] MW...")
        print(f"     For each P_G5, using multi-start to find feasibility (P_G1 = slack output)")

    # We also build a 2D grid by directly fixing both P_G5 and constraining P_G1 range
    # This requires a different approach: we CONSTRAIN P_G1 to a small range around a target
    # Actually the correct traditional approach is:
    # - For each (P_G5) value on the outer loop
    # - The inner loop fixes P_G1 (slack output) by adding an equality constraint
    # But slack bus by definition outputs whatever power needed for balance.
    #
    # The traditional code FIXES BOTH Pg[1] and Pg[2] simultaneously:
    #   m.Pg[2].fix(pg5_MW / baseMVA)  <- fixes G5
    #   m.Pg[1].fix(pg1_MW / baseMVA)  <- fixes G1 (slack!)
    # This means G1 is NO LONGER slack - the loads must sum to G1+G5 exactly
    # (with losses). If no feasible AC power flow exists for these exact values, infeasible.

    # So the 2D grid scans (P_G1, P_G5) with BOTH fixed simultaneously.
    # Outer loop: P_G5 ∈ [0, 400] step 2 MW
    # Inner loop: P_G1 ∈ [0, 700] step 5 MW (coarse), then refine

    n_pg5 = n_per_dim
    n_pg1 = n_per_dim

    labels_2d = np.zeros((n_pg5, n_pg1), dtype=np.float32)

    for i, pg5_MW in enumerate(pg5_arr):
        if verbose and i % 20 == 0:
            total_feas = labels_2d[:i].sum()
            print(f"  P_G5={pg5_MW:.0f} MW ({i}/{n_pg5}), total feasible so far: {int(total_feas)}")

        warm_pools = {
            ('inc', 'flat'): None, ('inc', 'outer'): None,
            ('dec', 'flat'): None, ('dec', 'outer'): None,
        }

        # Forward scan (increasing P_G1)
        for j, pg1_MW in enumerate(pg1_arr):
            pg_fixed = {1: pg1_MW / baseMVA, 2: pg5_MW / baseMVA}
            feasible, _, warm_pools = checker.check_multistart(pg_fixed, warm_pools, 'inc')
            if feasible:
                labels_2d[i, j] = 1.0

        # Reverse scan (decreasing P_G1) to catch disconnected components
        warm_pools = {
            ('inc', 'flat'): None, ('inc', 'outer'): None,
            ('dec', 'flat'): None, ('dec', 'outer'): None,
        }
        for j in range(n_pg1 - 1, -1, -1):
            pg1_MW = pg1_arr[j]
            if labels_2d[i, j] == 0:  # only re-check if not already feasible
                pg_fixed = {1: pg1_MW / baseMVA, 2: pg5_MW / baseMVA}
                feasible, _, warm_pools = checker.check_multistart(pg_fixed, warm_pools, 'dec')
                if feasible:
                    labels_2d[i, j] = 1.0

    # Flatten to 1D arrays for the dataset
    PG1_grid, PG5_grid = np.meshgrid(pg1_arr, pg5_arr)
    labels_flat = labels_2d.ravel().astype(np.float32)

    X = np.column_stack([PG1_grid.ravel(), PG5_grid.ravel()]).astype(np.float32)
    pg1_nom, pg5_nom = 350.0, 200.0
    X_mean = np.array([pg1_nom, pg5_nom], dtype=np.float32)
    X_std = np.array([350.0, 200.0], dtype=np.float32)
    X_norm = (X - X_mean) / X_std

    if verbose:
        print(f"WB5 2D grid: security rate = {labels_flat.mean():.3f}, "
              f"total feasible = {int(labels_flat.sum())}")

    return X_norm, labels_flat, {
        'name': 'WB5',
        'case': 'WB5',
        'PG1_grid': PG1_grid,
        'PG5_grid': PG5_grid,
        'pg1_arr': pg1_arr,
        'pg5_arr': pg5_arr,
        'labels_2d': labels_2d,
        'n_per_dim': n_per_dim,
        'X_mean': X_mean,
        'X_std': X_std,
        'pg1_range': pg1_range,
        'pg5_range': pg5_range,
        'axis_labels': ('P_G1 (MW)', 'P_G5 (MW)'),
        'security_rate': float(labels_flat.mean()),
        'space': 'generator_power_space',
    }


# ══════════════════════════════════════════════════════════════════
# case9mod System Configuration
# ══════════════════════════════════════════════════════════════════

def _get_case9mod_config() -> Dict:
    """case9mod system data in MATPOWER format (matching D:/安全域/1/case9线路热极限.py)."""
    baseMVA = 100.0

    bus = np.array([
        [1, 3,  0,  0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],   # slack
        [2, 2,  0,  0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [3, 2,  0,  0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [4, 1,  0,  0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [5, 1, 54, 18, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [6, 1,  0,  0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [7, 1, 60, 21, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [8, 1,  0,  0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        [9, 1, 75, 30, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
    ], dtype=float)

    # gen: [bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin]
    # Qmin = -5 MVAR (tightened from -300)
    gen = np.array([
        [1,   0, 0, 300, -5, 1, 100, 1, 250, 10],   # G1 (slack)
        [2, 163, 0, 300, -5, 1, 100, 1, 300, 10],   # G2
        [3,  85, 0, 300, -5, 1, 100, 1, 270, 10],   # G3
    ], dtype=float)

    branch = np.array([
        [1, 4, 0.0,    0.0576, 0,     250, 250, 250, 0, 0, 1, -360, 360],
        [4, 5, 0.017,  0.092,  0.158, 250, 250, 250, 0, 0, 1, -360, 360],
        [5, 6, 0.039,  0.17,   0.358, 150, 150, 150, 0, 0, 1, -360, 360],
        [3, 6, 0.0,    0.0586, 0,     300, 300, 300, 0, 0, 1, -360, 360],
        [6, 7, 0.0119, 0.1008, 0.209, 150, 150, 150, 0, 0, 1, -360, 360],
        [7, 8, 0.0085, 0.072,  0.149, 250, 250, 250, 0, 0, 1, -360, 360],
        [8, 2, 0.0,    0.0625, 0,     250, 250, 250, 0, 0, 1, -360, 360],
        [8, 9, 0.032,  0.161,  0.306, 250, 250, 250, 0, 0, 1, -360, 360],
        [9, 4, 0.01,   0.085,  0.176, 250, 250, 250, 0, 0, 1, -360, 360],
    ], dtype=float)

    return {
        'bus': bus, 'gen': gen, 'branch': branch,
        'baseMVA': baseMVA,
        'slack_bus': 1,
        'fixed_gen_indices': [2, 3],  # Fix G2 and G3; G1 is slack
        'outer_Va_deg': {1: 0, 2: 9.28, 3: 4.66, 4: -2.22, 5: -3.99,
                         6: -3.69, 7: -6.98, 8: -1.76, 9: -7.87},
        'solver_tol': 1e-8,
        'solver_max_iter': 3000,
    }


def generate_case9mod_data(
    n_samples: int = 8000,
    seed: int = 42,
    pg2_range: Tuple[float, float] = (10.0, 300.0),
    pg3_range: Tuple[float, float] = (10.0, 270.0),
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    case9mod: Generate SSR dataset in (P_G2, P_G3) space.
    Loads fixed at nominal: [54, 60, 75] MW at buses [5, 7, 9].
    P_G1 (slack) determined automatically.

    Expects 3 disconnected feasible components.
    """
    if not HAS_PYOMO:
        raise RuntimeError("Pyomo required.")

    cfg = _get_case9mod_config()
    checker = PyomoFeasibilityChecker(cfg)
    baseMVA = cfg['baseMVA']

    sampler = LatinHypercube(d=2, seed=seed)
    samples = sampler.random(n=n_samples)
    pg2_arr = pg2_range[0] + samples[:, 0] * (pg2_range[1] - pg2_range[0])
    pg3_arr = pg3_range[0] + samples[:, 1] * (pg3_range[1] - pg3_range[0])

    labels = np.zeros(n_samples, dtype=np.float32)

    if verbose:
        print(f"case9mod: Generating {n_samples} samples in "
              f"P_G2 ∈ [{pg2_range[0]:.0f},{pg2_range[1]:.0f}], "
              f"P_G3 ∈ [{pg3_range[0]:.0f},{pg3_range[1]:.0f}] MW...")

    warm_pools = {
        ('inc', 'flat'): None, ('inc', 'outer'): None,
        ('dec', 'flat'): None, ('dec', 'outer'): None,
    }

    for i in range(n_samples):
        if verbose and i % 1000 == 0:
            print(f"  {i}/{n_samples}, feas rate: {labels[:max(1,i)].mean():.3f}")
        pg_fixed = {
            2: pg2_arr[i] / baseMVA,
            3: pg3_arr[i] / baseMVA,
        }
        feasible, _, warm_pools = checker.check_multistart(pg_fixed, warm_pools, 'inc')
        labels[i] = float(feasible)

    X = np.column_stack([pg2_arr, pg3_arr]).astype(np.float32)
    pg2_nom, pg3_nom = 163.0, 85.0
    X_mean = np.array([pg2_nom, pg3_nom], dtype=np.float32)
    X_std = np.array([145.0, 130.0], dtype=np.float32)
    X_norm = (X - X_mean) / X_std

    if verbose:
        print(f"case9mod: security rate = {labels.mean():.3f}")

    return X_norm, labels, {
        'name': 'case9mod',
        'case': 'case9mod',
        'n_gen': 3,
        'n_bus': 9,
        'control_vars': ['P_G2 (MW)', 'P_G3 (MW)'],
        'feature_names': ['P_G2 (MW)', 'P_G3 (MW)'],
        'pg2_raw': pg2_arr,
        'pg3_raw': pg3_arr,
        'pg2_range': pg2_range,
        'pg3_range': pg3_range,
        'X_mean': X_mean,
        'X_std': X_std,
        'security_rate': float(labels.mean()),
        'space': 'generator_power_space',
    }


def generate_case9mod_grid(
    n_per_dim: int = 60,
    pg2_range: Tuple[float, float] = (10.0, 300.0),
    pg3_range: Tuple[float, float] = (10.0, 270.0),
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    case9mod: 2D grid in (P_G2, P_G3) space for visualization.
    Matches the traditional IPOPT scanning approach.
    """
    if not HAS_PYOMO:
        raise RuntimeError("Pyomo required.")

    cfg = _get_case9mod_config()
    checker = PyomoFeasibilityChecker(cfg)
    baseMVA = cfg['baseMVA']

    pg2_arr = np.linspace(pg2_range[0], pg2_range[1], n_per_dim)
    pg3_arr = np.linspace(pg3_range[0], pg3_range[1], n_per_dim)
    PG2, PG3 = np.meshgrid(pg2_arr, pg3_arr)
    n = n_per_dim ** 2
    labels = np.zeros(n, dtype=np.float32)
    pg1_results = np.full(n, float('nan'))

    if verbose:
        print(f"case9mod: {n_per_dim}x{n_per_dim} grid in (P_G2, P_G3) space...")

    pg2_flat = PG2.ravel()
    pg3_flat = PG3.ravel()

    warm_pools = {
        ('inc', 'flat'): None, ('inc', 'outer'): None,
        ('dec', 'flat'): None, ('dec', 'outer'): None,
    }

    for i in range(n):
        if verbose and i % 500 == 0:
            print(f"  {i}/{n}, feas rate: {labels[:max(1,i)].mean():.3f}")
        pg_fixed = {
            2: pg2_flat[i] / baseMVA,
            3: pg3_flat[i] / baseMVA,
        }
        feasible, warm_state, warm_pools = checker.check_multistart(
            pg_fixed, warm_pools, 'inc')
        labels[i] = float(feasible)

    # Also do reverse scan
    warm_pools = {k: None for k in warm_pools}
    for i in range(n - 1, -1, -1):
        if labels[i] == 0:
            pg_fixed = {
                2: pg2_flat[i] / baseMVA,
                3: pg3_flat[i] / baseMVA,
            }
            feasible, _, warm_pools = checker.check_multistart(
                pg_fixed, warm_pools, 'dec')
            labels[i] = float(feasible)

    X = np.column_stack([pg2_flat, pg3_flat]).astype(np.float32)
    pg2_nom, pg3_nom = 163.0, 85.0
    X_mean = np.array([pg2_nom, pg3_nom], dtype=np.float32)
    X_std = np.array([145.0, 130.0], dtype=np.float32)
    X_norm = (X - X_mean) / X_std

    if verbose:
        print(f"case9mod grid: security rate = {labels.mean():.3f}")

    return X_norm, labels, {
        'name': 'case9mod',
        'PG2_grid': PG2,
        'PG3_grid': PG3,
        'pg2_arr': pg2_arr,
        'pg3_arr': pg3_arr,
        'labels_2d': labels.reshape(n_per_dim, n_per_dim),
        'n_per_dim': n_per_dim,
        'X_mean': X_mean,
        'X_std': X_std,
        'pg2_range': pg2_range,
        'pg3_range': pg3_range,
        'axis_labels': ('P_G2 (MW)', 'P_G3 (MW)'),
        'security_rate': float(labels.mean()),
        'space': 'generator_power_space',
    }


# ══════════════════════════════════════════════════════════════════
# LMBM3 System Configuration
# ══════════════════════════════════════════════════════════════════

def _get_lmbm3_config(load_factor: float = 1.0) -> Dict:
    """
    LMBM3: 3-bus triangular case with binding line limit (L3-2 = 186 MVA).
    Scans (P_G1, P_G2) space with P_G3 fixed at 0 (or varied).
    load_factor scales all loads uniformly.
    """
    baseMVA = 100.0
    lf = load_factor

    bus = np.array([
        [1, 3, 110 * lf, 40 * lf, 0, 0, 1, 1.069, 0, 345, 1, 1.1, 0.9],
        [2, 2, 110 * lf, 40 * lf, 0, 0, 1, 1.028, 0, 345, 1, 1.1, 0.9],
        [3, 1,  95 * lf, 50 * lf, 0, 0, 1, 1.001, 0, 345, 1, 1.1, 0.9],
    ], dtype=float)

    # G1=slack, G2=PV, G3=PV (but Pmax=0 for G3 in nominal)
    gen = np.array([
        [1, 0,    0, 10000, -1000, 1.069, 100, 1, 10000, 0],
        [2, 185.93, 0, 1000, -1000, 1.028, 100, 1, 10000, 0],
        [3, 0,    0, 1000, -1000, 1.001, 100, 1, 10000, 0],
    ], dtype=float)

    # Line 3-2 has rateA=186 MVA; others unconstrained
    branch = np.array([
        [1, 3, 0.065, 0.620, 0, 9999, 9999, 9999, 0, 0, 1, -360, 360],  # L1-3
        [3, 2, 0.025, 0.750, 0,  186,  186,  186, 0, 0, 1, -360, 360],  # L3-2 binding!
        [1, 2, 0.042, 0.900, 0, 9999, 9999, 9999, 0, 0, 1, -360, 360],  # L1-2
    ], dtype=float)

    return {
        'bus': bus, 'gen': gen, 'branch': branch,
        'baseMVA': baseMVA,
        'slack_bus': 1,
        'fixed_gen_indices': [2],  # Fix G2; G1 is slack; G3 is PV with Pmax=0
        'outer_Va_deg': {1: 0.0, 2: -5.0, 3: -10.0},
        'solver_tol': 1e-8,
        'solver_max_iter': 3000,
        'load_factor': load_factor,
    }


def generate_LMBM3_data(
    n_samples: int = 5000,
    seed: int = 42,
    load_factor: float = 1.0,
    pg2_range: Tuple[float, float] = (0.0, 600.0),
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    LMBM3: Generate SSR dataset sweeping P_G2 with loads at load_factor * nominal.
    P_G1 (slack) determined automatically. L3-2 line limit (186 MVA) creates disconnected region.
    """
    if not HAS_PYOMO:
        raise RuntimeError("Pyomo required.")

    cfg = _get_lmbm3_config(load_factor)
    checker = PyomoFeasibilityChecker(cfg)
    baseMVA = cfg['baseMVA']

    sampler = LatinHypercube(d=1, seed=seed)
    samples = sampler.random(n=n_samples)
    pg2_arr = pg2_range[0] + samples[:, 0] * (pg2_range[1] - pg2_range[0])

    labels = np.zeros(n_samples, dtype=np.float32)

    if verbose:
        print(f"LMBM3 (λ={load_factor}): Generating {n_samples} samples "
              f"in P_G2 ∈ [{pg2_range[0]:.0f},{pg2_range[1]:.0f}] MW...")

    warm_pools = {
        ('inc', 'flat'): None, ('inc', 'outer'): None,
        ('dec', 'flat'): None, ('dec', 'outer'): None,
    }

    for i, pg2_MW in enumerate(pg2_arr):
        if verbose and i % 500 == 0:
            print(f"  {i}/{n_samples}, feas rate: {labels[:max(1,i)].mean():.3f}")
        pg_fixed = {2: pg2_MW / baseMVA}
        feasible, _, warm_pools = checker.check_multistart(pg_fixed, warm_pools, 'inc')
        labels[i] = float(feasible)

    X = pg2_arr[:, None].astype(np.float32)
    pg2_nom = 185.93
    X_mean = np.array([pg2_nom], dtype=np.float32)
    X_std = np.array([300.0], dtype=np.float32)
    X_norm = (X - X_mean) / X_std

    if verbose:
        print(f"LMBM3 (λ={load_factor}): security rate = {labels.mean():.3f}")

    return X_norm, labels, {
        'name': f'LMBM3_lf{load_factor:.2f}',
        'case': 'LMBM3',
        'load_factor': load_factor,
        'n_gen': 3,
        'n_bus': 3,
        'control_vars': ['P_G2 (MW)'],
        'feature_names': ['P_G2 (MW)'],
        'pg2_raw': pg2_arr,
        'pg2_range': pg2_range,
        'X_mean': X_mean,
        'X_std': X_std,
        'security_rate': float(labels.mean()),
        'space': 'generator_power_space',
    }


# ══════════════════════════════════════════════════════════════════
# Load pre-computed traditional IPOPT results from CSV
# ══════════════════════════════════════════════════════════════════

def load_traditional_results(case_name: str, data_dir: str = r'D:\安全域\1') -> Optional[Dict]:
    """
    Load the pre-computed traditional IPOPT results to use as ground truth
    for comparison and as training data supplement.
    """
    import os
    try:
        import pandas as pd
    except ImportError:
        print("pandas required for loading traditional results")
        return None

    file_map = {
        'WB5': os.path.join(data_dir, '5节点数据.csv'),
        'case9mod': os.path.join(data_dir, 'ac_opf_9results.csv'),
        'LMBM3_1.5': os.path.join(data_dir, 'lmbm3_feasible_points_v2_optimized.csv'),
        'LMBM3_1.49': os.path.join(data_dir, 'lmbm3 负荷1.490.csv'),
    }

    fname = file_map.get(case_name)
    if fname is None or not os.path.exists(fname):
        print(f"Traditional results not found for {case_name} at {fname}")
        return None

    try:
        df = pd.read_csv(fname)
        print(f"Loaded {len(df)} traditional IPOPT results for {case_name} from {fname}")
        return {'df': df, 'file': fname, 'case': case_name}
    except Exception as e:
        print(f"Error loading {fname}: {e}")
        return None


def get_wb5_traditional_dataset(
    data_dir: str = r'D:\安全域\1',
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load WB5 traditional results and convert to (X, y) format in generator power space.
    Traditional data: columns [PG5, PG1, loss] - all are FEASIBLE points.
    We need to add insecure points by creating a background grid.
    """
    trad = load_traditional_results('WB5', data_dir)
    if trad is None:
        return None, None, {}

    df = trad['df']
    # Traditional data: PG5 (Y-axis), PG1 (X-axis) - all feasible
    pg1_feas = df['PG1'].values if 'PG1' in df.columns else df.iloc[:, 1].values
    pg5_feas = df['PG5'].values if 'PG5' in df.columns else df.iloc[:, 0].values

    # Build exact IPOPT scan lattice and sample background from the complement.
    # WB5 traditional scans: PG1 step=0.5 MW, PG5 step=1.0 MW.
    pg1_vals = np.arange(0.0, 700.0 + 0.5, 0.5, dtype=np.float32)
    pg5_vals = np.arange(0.0, 400.0 + 1.0, 1.0, dtype=np.float32)

    secure_mask = np.zeros((len(pg5_vals), len(pg1_vals)), dtype=bool)
    ix = np.rint((pg1_feas - pg1_vals[0]) / 0.5).astype(int)
    iy = np.rint((pg5_feas - pg5_vals[0]) / 1.0).astype(int)
    valid = (ix >= 0) & (ix < len(pg1_vals)) & (iy >= 0) & (iy < len(pg5_vals))
    secure_mask[iy[valid], ix[valid]] = True

    # Secure points
    X_feas = np.column_stack([pg1_feas, pg5_feas]).astype(np.float32)
    y_feas = np.ones(len(X_feas), dtype=np.float32)

    # Insecure background points: sample exact-grid complement
    insecure_idx = np.argwhere(~secure_mask)
    n_bg = min(len(X_feas) * 2, len(insecure_idx))
    rng = np.random.default_rng(seed)
    pick = insecure_idx[rng.choice(len(insecure_idx), size=n_bg, replace=False)]
    X_bg = np.column_stack([pg1_vals[pick[:, 1]], pg5_vals[pick[:, 0]]]).astype(np.float32)
    y_bg = np.zeros(len(X_bg), dtype=np.float32)

    X_all = np.vstack([X_feas, X_bg])
    y_all = np.concatenate([y_feas, y_bg])

    X_mean = np.array([350.0, 200.0], dtype=np.float32)
    X_std = np.array([350.0, 200.0], dtype=np.float32)
    X_norm = (X_all - X_mean) / X_std

    meta = {
        'name': 'WB5',
        'source': 'traditional+exact_grid_background',
        'n_feasible': len(X_feas),
        'n_infeasible': len(X_bg),
        'X_mean': X_mean,
        'X_std': X_std,
        'pg1_feas': pg1_feas,
        'pg5_feas': pg5_feas,
        'feature_names': ['P_G1 (MW)', 'P_G5 (MW)'],
        'security_rate': float(y_all.mean()),
        'space': 'generator_power_space',
        'grid_shape': [int(len(pg5_vals)), int(len(pg1_vals))],
    }
    print(f"WB5 traditional dataset: {len(X_feas)} secure + {len(X_bg)} insecure-grid background = {len(X_all)} total")
    return X_norm, y_all, meta


def get_case9mod_traditional_dataset(
    data_dir: str = r'D:\安全域\1',
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load case9mod traditional results and convert to (X, y) format.
    Traditional data: columns [p1_mw, p2_mw, p3_mw, ...] - all FEASIBLE.
    """
    trad = load_traditional_results('case9mod', data_dir)
    if trad is None:
        return None, None, {}

    df = trad['df']
    pg1_feas = df['p1_mw'].values
    pg2_feas = df['p2_mw'].values
    pg3_feas = df['p3_mw'].values

    # Feasible points in (P_G2, P_G3) space (P_G1 = slack, not a free variable)
    X_feas = np.column_stack([pg2_feas, pg3_feas]).astype(np.float32)
    y_feas = np.ones(len(X_feas), dtype=np.float32)

    # Build exact IPOPT scan lattice and sample background from the complement.
    # case9mod traditional scans use 300 points on each axis.
    pg2_vals = np.linspace(10.0, 300.0, 300, dtype=np.float32)
    pg3_vals = np.linspace(10.0, 270.0, 300, dtype=np.float32)
    dpg2 = float(pg2_vals[1] - pg2_vals[0])
    dpg3 = float(pg3_vals[1] - pg3_vals[0])

    secure_mask = np.zeros((len(pg3_vals), len(pg2_vals)), dtype=bool)
    ix = np.rint((pg2_feas - pg2_vals[0]) / dpg2).astype(int)
    iy = np.rint((pg3_feas - pg3_vals[0]) / dpg3).astype(int)
    valid = (ix >= 0) & (ix < len(pg2_vals)) & (iy >= 0) & (iy < len(pg3_vals))
    secure_mask[iy[valid], ix[valid]] = True

    insecure_idx = np.argwhere(~secure_mask)
    n_bg = min(len(X_feas) * 2, len(insecure_idx))
    rng = np.random.default_rng(seed)
    pick = insecure_idx[rng.choice(len(insecure_idx), size=n_bg, replace=False)]
    X_bg_grid = np.column_stack([pg2_vals[pick[:, 1]], pg3_vals[pick[:, 0]]]).astype(np.float32)
    y_bg_grid = np.zeros(n_bg, dtype=np.float32)

    # Domain-guard negatives: explicit hard negatives below lower active-power
    # limits (e.g., 0-10 MW), which are physically outside the valid scan region.
    # This prevents out-of-support false positives in visualization.
    pg2_floor = float(pg2_vals[0])
    pg3_floor = float(pg3_vals[0])
    vis_hi = 180.0
    n_guard = min(len(X_feas), 3000)
    n_guard_x = n_guard // 2
    n_guard_y = n_guard - n_guard_x

    X_guard_x = np.column_stack([
        rng.uniform(0.0, pg2_floor, size=n_guard_x),
        rng.uniform(0.0, vis_hi, size=n_guard_x),
    ]).astype(np.float32)
    X_guard_y = np.column_stack([
        rng.uniform(0.0, vis_hi, size=n_guard_y),
        rng.uniform(0.0, pg3_floor, size=n_guard_y),
    ]).astype(np.float32)
    X_guard = np.vstack([X_guard_x, X_guard_y]).astype(np.float32)
    y_guard = np.zeros(len(X_guard), dtype=np.float32)

    X_bg = np.vstack([X_bg_grid, X_guard]).astype(np.float32)
    y_bg = np.concatenate([y_bg_grid, y_guard]).astype(np.float32)

    X_all = np.vstack([X_feas, X_bg])
    y_all = np.concatenate([y_feas, y_bg])

    X_mean = np.array([163.0, 85.0], dtype=np.float32)
    X_std = np.array([145.0, 130.0], dtype=np.float32)
    X_norm = (X_all - X_mean) / X_std

    meta = {
        'name': 'case9mod',
        'source': 'traditional+exact_grid_background+domain_guard_negatives',
        'n_feasible': len(X_feas),
        'n_infeasible': len(X_bg),
        'n_bg_grid': int(len(X_bg_grid)),
        'n_bg_guard': int(len(X_guard)),
        'X_mean': X_mean,
        'X_std': X_std,
        'pg1_feas': pg1_feas,
        'pg2_feas': pg2_feas,
        'pg3_feas': pg3_feas,
        'feature_names': ['P_G2 (MW)', 'P_G3 (MW)'],
        'security_rate': float(y_all.mean()),
        'space': 'generator_power_space',
        'n_components': 3,
        'grid_shape': [int(len(pg3_vals)), int(len(pg2_vals))],
    }
    print(
        "case9mod traditional dataset: "
        f"{len(X_feas)} secure + {len(X_bg_grid)} grid-insecure + {len(X_guard)} domain-guard "
        f"= {len(X_all)} total"
    )
    return X_norm, y_all, meta


# ══════════════════════════════════════════════════════════════════
# Unified interface
# ══════════════════════════════════════════════════════════════════

def generate_ssr_data(
    case_name: str,
    n_samples: int = 5000,
    seed: int = 42,
    use_traditional: bool = True,
    data_dir: str = r'D:\安全域\1',
    verbose: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Unified data generation interface in GENERATOR POWER SPACE.

    Preference order:
    1. Load pre-computed traditional IPOPT results (fastest, most accurate)
    2. Generate new data using Pyomo+IPOPT (slow but correct)

    All cases work in GENERATOR POWER SPACE (P_Gi as axes, loads fixed).
    """
    if case_name == 'WB2':
        return generate_WB2_data(n_samples, seed)

    # Try loading traditional results first
    if use_traditional:
        if case_name == 'WB5':
            X, y, meta = get_wb5_traditional_dataset(data_dir, seed=seed)
            if X is not None:
                return X, y, meta
        elif case_name == 'case9mod':
            X, y, meta = get_case9mod_traditional_dataset(data_dir, seed=seed)
            if X is not None:
                return X, y, meta

    # Fall back to Pyomo generation
    if case_name == 'WB5':
        return generate_WB5_data(n_samples, seed, verbose=verbose, **kwargs)
    elif case_name == 'case9mod':
        return generate_case9mod_data(n_samples, seed, verbose=verbose, **kwargs)
    elif case_name == 'LMBM3':
        lf = kwargs.get('load_factor', 1.0)
        return generate_LMBM3_data(n_samples, seed, load_factor=lf, verbose=verbose)
    else:
        raise ValueError(f"Unknown case: {case_name}. Available: WB2, WB5, case9mod, LMBM3")


def generate_ssr_grid(
    case_name: str,
    n_per_dim: int = 60,
    data_dir: str = r'D:\安全域\1',
    verbose: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Generate 2D grid for visualization in generator power space."""
    if case_name == 'WB2':
        return generate_WB2_grid(n_per_dim)
    elif case_name == 'WB5':
        return generate_WB5_grid(n_per_dim, verbose=verbose, **kwargs)
    elif case_name == 'case9mod':
        return generate_case9mod_grid(n_per_dim, verbose=verbose, **kwargs)
    else:
        raise ValueError(f"Grid generation not supported for {case_name}")


if __name__ == '__main__':
    print("Testing data generation in generator power space...")
    print()

    # Test WB2 (analytical)
    print("=== WB2 (analytical) ===")
    X, y, meta = generate_WB2_data(n_samples=500)
    print(f"  Shape: {X.shape}, security: {y.mean():.3f}")

    # Test loading traditional WB5 results
    print("\n=== WB5 (traditional IPOPT results) ===")
    X, y, meta = generate_ssr_data('WB5', use_traditional=True)
    if X is not None:
        print(f"  Shape: {X.shape}, security: {y.mean():.3f}")

    # Test loading traditional case9mod results
    print("\n=== case9mod (traditional IPOPT results) ===")
    X, y, meta = generate_ssr_data('case9mod', use_traditional=True)
    if X is not None:
        print(f"  Shape: {X.shape}, security: {y.mean():.3f}")
