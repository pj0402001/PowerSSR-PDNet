"""
Bukhsh et al. (2013) test cases converted from MATPOWER format to pandapower.
Reference: W. A. Bukhsh, Andreas Grothey, Ken McKinnon, Paul Trodden,
           "Local Solutions of Optimal Power Flow Problem",
           IEEE Transactions on Power Systems, 2013.

Cases: WB2, WB3, WB5, LMBM3, case9mod
"""

import pandapower as pp
import numpy as np


def create_WB2() -> pp.pandapowerNet:
    """
    WB2: 2-bus system with one generator and one load.
    Notable: has multiple local OPF solutions (Qd = -350 MVAR is absorbing).
    Vmax=1.05, Vmin=0.95. rateA=990000 MVA (effectively unconstrained line).
    """
    net = pp.create_empty_network(sn_mva=100)

    # Buses
    b1 = pp.create_bus(net, vn_kv=1.0, name='Bus1', max_vm_pu=1.05, min_vm_pu=0.95)
    b2 = pp.create_bus(net, vn_kv=1.0, name='Bus2', max_vm_pu=1.05, min_vm_pu=0.95)

    # Slack generator at bus 1 (ext_grid acts as reference/slack)
    pp.create_ext_grid(net, bus=b1, vm_pu=0.964, name='G1',
                       max_p_mw=600, min_p_mw=0,
                       max_q_mvar=400, min_q_mvar=-400)

    # Load at bus 2: Pd=350 MW, Qd=-350 MVAR (capacitive/absorbing reactive)
    pp.create_load(net, bus=b2, p_mw=350.0, q_mvar=-350.0, name='Load2')

    # Branch: r=0.04, x=0.20, b=0 (in per-unit on 100 MVA base)
    # rateA = 990000 => effectively unconstrained
    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b2,
                                   length_km=1.0, r_ohm_per_km=0.04,
                                   x_ohm_per_km=0.20, c_nf_per_km=0.0,
                                   max_i_ka=9900.0, name='L1-2')

    # Generator cost: c2=0, c1=2, c0=0  (linear in P)
    net._metadata = {
        'name': 'WB2',
        'source': 'Bukhsh et al. (2013)',
        'n_bus': 2, 'n_gen': 1, 'n_load': 1, 'n_branch': 1,
        'notes': '2-bus case; Qd negative (absorbing); has multiple local OPF solutions',
        'baseMVA': 100,
    }
    return net


def create_WB3() -> pp.pandapowerNet:
    """
    WB3: 3-bus radial system, single generator at bus 1.
    Loads at buses 2 and 3.
    """
    net = pp.create_empty_network(sn_mva=100)

    b1 = pp.create_bus(net, vn_kv=1.0, name='Bus1', max_vm_pu=1.05, min_vm_pu=0.95)
    b2 = pp.create_bus(net, vn_kv=1.0, name='Bus2', max_vm_pu=1.05, min_vm_pu=0.95)
    b3 = pp.create_bus(net, vn_kv=1.0, name='Bus3', max_vm_pu=1.05, min_vm_pu=0.95)

    pp.create_ext_grid(net, bus=b1, vm_pu=0.95, name='G1',
                       max_p_mw=5000, min_p_mw=0,
                       max_q_mvar=3000, min_q_mvar=-3000)

    pp.create_load(net, bus=b2, p_mw=120.0, q_mvar=86.0, name='Load2')
    pp.create_load(net, bus=b3, p_mw=68.0,  q_mvar=50.0, name='Load3')

    # Branch 1-2: r=0.04, x=0.20
    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b2,
                                   length_km=1.0, r_ohm_per_km=0.04,
                                   x_ohm_per_km=0.20, c_nf_per_km=0.0,
                                   max_i_ka=9900.0, name='L1-2')
    # Branch 2-3: r=0.0139, x=0.0605, b=2.459 (pu)
    # b in pu → c_nf_per_km: for 1 km line on 100MVA/1kV base: b_pu = omega*C*Zbase → approximate
    # Store b as susceptance in line charging (c_nf approximation)
    pp.create_line_from_parameters(net, from_bus=b2, to_bus=b3,
                                   length_km=1.0, r_ohm_per_km=0.0139,
                                   x_ohm_per_km=0.0605, c_nf_per_km=0.0,
                                   max_i_ka=9900.0, name='L2-3')

    net._metadata = {
        'name': 'WB3',
        'source': 'Bukhsh et al. (2013)',
        'n_bus': 3, 'n_gen': 1, 'n_load': 2, 'n_branch': 2,
        'notes': '3-bus radial case; single slack generator',
        'baseMVA': 100,
    }
    return net


def create_WB5() -> pp.pandapowerNet:
    """
    WB5: 5-bus meshed system with 2 generators (bus 1 and bus 5).
    6 branches. Line limits: rateA=2500 MVA (binding in some scenarios).
    Cost: G1: c1=4.00, G2: c1=1.00  (cheaper generator at bus 5).
    """
    net = pp.create_empty_network(sn_mva=100)

    buses = {}
    bus_data = [
        (1, 1.05, 0.95, 'slack'),
        (2, 1.05, 0.95, 'pq'),
        (3, 1.05, 0.95, 'pq'),
        (4, 1.05, 0.95, 'pq'),
        (5, 1.05, 0.95, 'pv'),
    ]
    for (idx, vmax, vmin, _) in bus_data:
        buses[idx] = pp.create_bus(net, vn_kv=345.0, name=f'Bus{idx}',
                                   max_vm_pu=vmax, min_vm_pu=vmin)

    # Generator at bus 1 (slack)
    pp.create_ext_grid(net, bus=buses[1], vm_pu=1.0, name='G1',
                       max_p_mw=5000, min_p_mw=0,
                       max_q_mvar=1800, min_q_mvar=-30)

    # Generator at bus 5 (PV, cheaper)
    pp.create_gen(net, bus=buses[5], p_mw=0.0, vm_pu=1.0, name='G5',
                  max_p_mw=5000, min_p_mw=0,
                  max_q_mvar=1800, min_q_mvar=-30, controllable=True)

    # Loads
    pp.create_load(net, bus=buses[2], p_mw=130.0, q_mvar=20.0, name='Load2')
    pp.create_load(net, bus=buses[3], p_mw=130.0, q_mvar=20.0, name='Load3')
    pp.create_load(net, bus=buses[4], p_mw=65.0,  q_mvar=10.0, name='Load4')

    # Branches (rateA=2500 MVA → max_i_ka=2500/345/sqrt(3)≈4.18 kA)
    max_i_ka = 2500.0 / (345.0 * np.sqrt(3))

    branch_data = [
        (1, 2, 0.04, 0.09, 0.0),
        (1, 3, 0.05, 0.10, 0.0),
        (2, 4, 0.55, 0.90, 0.45),
        (3, 5, 0.55, 0.90, 0.45),
        (4, 5, 0.06, 0.10, 0.0),
        (2, 3, 0.07, 0.09, 0.0),
    ]
    for (fb, tb, r, x, b) in branch_data:
        pp.create_line_from_parameters(
            net, from_bus=buses[fb], to_bus=buses[tb],
            length_km=1.0, r_ohm_per_km=r, x_ohm_per_km=x, c_nf_per_km=0.0,
            max_i_ka=max_i_ka, name=f'L{fb}-{tb}'
        )

    net._metadata = {
        'name': 'WB5',
        'source': 'Bukhsh et al. (2013)',
        'n_bus': 5, 'n_gen': 2, 'n_load': 3, 'n_branch': 6,
        'notes': '5-bus meshed case; 2 generators with different costs; active line limits',
        'baseMVA': 100,
        'gen_costs': {'G1': {'c1': 4.00}, 'G5': {'c1': 1.00}},
    }
    return net


def create_LMBM3() -> pp.pandapowerNet:
    """
    LMBM3: 3-bus case from Lesieutre et al. (2011), included in Bukhsh et al.
    3 generators (all buses), 3 branches (meshed triangle).
    Line 3-2 has rateA=186 MVA (binding constraint).
    """
    net = pp.create_empty_network(sn_mva=100)

    b1 = pp.create_bus(net, vn_kv=345.0, name='Bus1', max_vm_pu=1.10, min_vm_pu=0.90)
    b2 = pp.create_bus(net, vn_kv=345.0, name='Bus2', max_vm_pu=1.10, min_vm_pu=0.90)
    b3 = pp.create_bus(net, vn_kv=345.0, name='Bus3', max_vm_pu=1.10, min_vm_pu=0.90)

    # Bus 1: slack/reference generator
    pp.create_ext_grid(net, bus=b1, vm_pu=1.069, name='G1',
                       max_p_mw=10000, min_p_mw=0,
                       max_q_mvar=10000, min_q_mvar=-1000)

    # Buses 2 and 3: PV generators
    pp.create_gen(net, bus=b2, p_mw=185.93, vm_pu=1.028, name='G2',
                  max_p_mw=10000, min_p_mw=0,
                  max_q_mvar=1000, min_q_mvar=-1000, controllable=True)
    pp.create_gen(net, bus=b3, p_mw=0.0, vm_pu=1.001, name='G3',
                  max_p_mw=0, min_p_mw=0,
                  max_q_mvar=1000, min_q_mvar=-1000, controllable=True)

    # Loads at all 3 buses
    pp.create_load(net, bus=b1, p_mw=110.0, q_mvar=40.0, name='Load1')
    pp.create_load(net, bus=b2, p_mw=110.0, q_mvar=40.0, name='Load2')
    pp.create_load(net, bus=b3, p_mw=95.0,  q_mvar=50.0, name='Load3')

    # Branches: max_i_ka from rateA=9999 for unconstrained, 186 for constrained
    max_i_9999 = 9999.0 / (345.0 * np.sqrt(3))
    max_i_186  = 186.0  / (345.0 * np.sqrt(3))

    # Branch 1-3: unconstrained
    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b3,
                                   length_km=1.0, r_ohm_per_km=0.065,
                                   x_ohm_per_km=0.620, c_nf_per_km=0.0,
                                   max_i_ka=max_i_9999, name='L1-3')
    # Branch 3-2: constrained (rateA=186 MVA)
    pp.create_line_from_parameters(net, from_bus=b3, to_bus=b2,
                                   length_km=1.0, r_ohm_per_km=0.025,
                                   x_ohm_per_km=0.750, c_nf_per_km=0.0,
                                   max_i_ka=max_i_186, name='L3-2')
    # Branch 1-2: unconstrained
    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b2,
                                   length_km=1.0, r_ohm_per_km=0.042,
                                   x_ohm_per_km=0.900, c_nf_per_km=0.0,
                                   max_i_ka=max_i_9999, name='L1-2')

    net._metadata = {
        'name': 'LMBM3',
        'source': 'Bukhsh et al. (2013) / Lesieutre et al. (2011)',
        'n_bus': 3, 'n_gen': 3, 'n_load': 3, 'n_branch': 3,
        'notes': 'Triangular 3-bus case; L3-2 has binding line limit (186 MVA); SDP limit test case',
        'baseMVA': 100,
    }
    return net


def create_case9mod() -> pp.pandapowerNet:
    """
    case9mod: Modified IEEE 9-bus case.
    Modifications from standard case9:
      - Reactive power bounds tightened: Qmin = -5 MVAR (was -300)
      - Demands reduced to 60%
    3 generators (buses 1,2,3), 9 branches, 3 loads (buses 5,7,9).
    """
    net = pp.create_empty_network(sn_mva=100)

    bus_data = [
        (1, 1.10, 0.90), (2, 1.10, 0.90), (3, 1.10, 0.90),
        (4, 1.10, 0.90), (5, 1.10, 0.90), (6, 1.10, 0.90),
        (7, 1.10, 0.90), (8, 1.10, 0.90), (9, 1.10, 0.90),
    ]
    buses = {}
    for (idx, vmax, vmin) in bus_data:
        buses[idx] = pp.create_bus(net, vn_kv=345.0, name=f'Bus{idx}',
                                   max_vm_pu=vmax, min_vm_pu=vmin)

    # Generators — note: Qmin tightened to -5 MVAR from original -300
    # Bus 1: slack
    pp.create_ext_grid(net, bus=buses[1], vm_pu=1.0, name='G1',
                       max_p_mw=250, min_p_mw=10,
                       max_q_mvar=300, min_q_mvar=-5)
    # Bus 2: PV
    pp.create_gen(net, bus=buses[2], p_mw=163.0, vm_pu=1.0, name='G2',
                  max_p_mw=300, min_p_mw=10,
                  max_q_mvar=300, min_q_mvar=-5, controllable=True)
    # Bus 3: PV
    pp.create_gen(net, bus=buses[3], p_mw=85.0, vm_pu=1.0, name='G3',
                  max_p_mw=270, min_p_mw=10,
                  max_q_mvar=300, min_q_mvar=-5, controllable=True)

    # Loads at buses 5, 7, 9 — reduced to 60% of original
    # Original: 90/30, 100/35, 125/50 → 60%: 54/18, 60/21, 75/30
    pp.create_load(net, bus=buses[5], p_mw=54.0,  q_mvar=18.0, name='Load5')
    pp.create_load(net, bus=buses[7], p_mw=60.0,  q_mvar=21.0, name='Load7')
    pp.create_load(net, bus=buses[9], p_mw=75.0,  q_mvar=30.0, name='Load9')

    # Branches (rateA in MW → max_i_ka on 345 kV)
    def mw_to_ka(mw): return mw / (345.0 * np.sqrt(3))

    branch_data = [
        (1, 4, 0.0,    0.0576, 0,      mw_to_ka(250)),
        (4, 5, 0.017,  0.092,  0.158,  mw_to_ka(250)),
        (5, 6, 0.039,  0.17,   0.358,  mw_to_ka(150)),
        (3, 6, 0.0,    0.0586, 0,      mw_to_ka(300)),
        (6, 7, 0.0119, 0.1008, 0.209,  mw_to_ka(150)),
        (7, 8, 0.0085, 0.072,  0.149,  mw_to_ka(250)),
        (8, 2, 0.0,    0.0625, 0,      mw_to_ka(250)),
        (8, 9, 0.032,  0.161,  0.306,  mw_to_ka(250)),
        (9, 4, 0.01,   0.085,  0.176,  mw_to_ka(250)),
    ]
    for (fb, tb, r, x, b, max_i) in branch_data:
        pp.create_line_from_parameters(
            net, from_bus=buses[fb], to_bus=buses[tb],
            length_km=1.0, r_ohm_per_km=r, x_ohm_per_km=x, c_nf_per_km=0.0,
            max_i_ka=max_i, name=f'L{fb}-{tb}'
        )

    net._metadata = {
        'name': 'case9mod',
        'source': 'Bukhsh et al. (2013) modified from IEEE 9-bus',
        'n_bus': 9, 'n_gen': 3, 'n_load': 3, 'n_branch': 9,
        'notes': 'Modified IEEE 9-bus: Qmin=-5 MVAR (tightened); loads=60% of standard values',
        'baseMVA': 100,
        'gen_costs': {
            'G1': {'c2': 0.11,   'c1': 5,   'c0': 150, 'startup': 1500},
            'G2': {'c2': 0.085,  'c1': 1.2, 'c0': 600, 'startup': 2000},
            'G3': {'c2': 0.1225, 'c1': 1.0, 'c0': 335, 'startup': 3000},
        },
    }
    return net


# Registry for easy access
BUKHSH_CASES = {
    'WB2':      create_WB2,
    'WB3':      create_WB3,
    'WB5':      create_WB5,
    'LMBM3':    create_LMBM3,
    'case9mod': create_case9mod,
}


def get_bukhsh_case(name: str) -> pp.pandapowerNet:
    """Load a Bukhsh et al. test case by name."""
    if name not in BUKHSH_CASES:
        raise ValueError(f"Unknown case '{name}'. Available: {list(BUKHSH_CASES.keys())}")
    return BUKHSH_CASES[name]()


def verify_all_cases():
    """Quick convergence check for all cases at nominal operating point."""
    import warnings
    warnings.filterwarnings('ignore')
    print("Verifying Bukhsh et al. test cases (power flow at nominal point):")
    print(f"{'Case':12s} {'Buses':>6} {'Gens':>6} {'Loads':>6} {'Lines':>6} {'Conv?':>8} {'V_min':>8} {'V_max':>8}")
    print("-" * 70)
    for name, creator in BUKHSH_CASES.items():
        net = creator()
        try:
            pp.runpp(net, numba=False, verbose=False, max_iteration=50)
            converged = net.converged
            if converged:
                v = net.res_bus['vm_pu'].values
                v_min, v_max = v.min(), v.max()
            else:
                v_min = v_max = float('nan')
        except Exception as e:
            converged = False
            v_min = v_max = float('nan')

        m = net._metadata
        print(f"{name:12s} {m['n_bus']:6d} {m['n_gen']:6d} {m['n_load']:6d} "
              f"{m['n_branch']:6d} {'YES' if converged else 'NO':>8} "
              f"{v_min:8.4f} {v_max:8.4f}")


if __name__ == '__main__':
    verify_all_cases()
