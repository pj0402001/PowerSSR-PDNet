# PowerSSR-PDNet: Primal-Dual Network for Power System Static Security Region Characterization

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Official implementation** of:

> **Deep Learning-Based Characterization of Power System Static Security Regions**
> Using Bukhsh et al. (2013) Multi-Solution OPF Test Cases

---

## Overview

This project applies deep learning to characterize the **Static Security Region (SSR)** of power systems — the set of all load/generation operating points for which the power flow equations have a feasible solution satisfying all operating constraints (voltage limits, line flow limits, generator limits).

We use the publicly available test cases from:

> W. A. Bukhsh, A. Grothey, K. McKinnon, P. Trodden, "Local Solutions of Optimal Power Flow Problem," *IEEE Transactions on Power Systems*, 2013.

These cases are specifically designed to exhibit **multiple local OPF solutions**, making security region characterization challenging and interesting.

## Test Cases

| Case     | Buses | Gens | Loads | Lines | Key Feature |
|----------|-------|------|-------|-------|-------------|
| WB2      | 2     | 1    | 1     | 1     | 2 local solutions; analytical power flow |
| WB3      | 3     | 1    | 2     | 2     | Radial; multiple PF solutions |
| WB5      | 5     | 2    | 3     | 6     | Meshed; 2 generators with different costs |
| LMBM3    | 3     | 3    | 3     | 3     | From Lesieutre et al.; binding line limit |
| case9mod | 9     | 3    | 3     | 9     | Modified IEEE 9-bus; tightened Q limits |

## Models

Three neural network architectures are compared:

### 1. Baseline NN
Standard feedforward classifier with focal loss for class imbalance.

### 2. Physics-NN
Adds voltage profile prediction branch and constraint violation penalty.

### 3. SSR-PDNet (Proposed)
Our proposed architecture featuring:
- **Dual-branch architecture**: shared feature extractor + classifier head + physics head
- **Lagrange dual training**: learnable dual variables for soft constraint enforcement
- **Contrastive boundary loss**: encourages sharp, well-defined security boundary
- **Residual connections**: for improved gradient flow

## Installation

```bash
# Clone repository
git clone https://github.com/pj0402001/PowerSSR-PDNet.git
cd PowerSSR-PDNet

# Create conda environment (recommended)
conda create -n ssr-pdnet python=3.10 -y
conda activate ssr-pdnet

# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
torch>=2.0.0
pandapower>=2.13
numpy>=1.24
scipy>=1.10
scikit-learn>=1.2
matplotlib>=3.7
seaborn>=0.12
tqdm>=4.65
pandas>=2.0
```

## Quick Start

```bash
# Run experiment on WB2 (fast, analytical)
python experiments/run_bukhsh.py --cases WB2

# Run all main test cases (WB2 + case9mod)
python experiments/run_bukhsh.py --cases WB2 WB5 case9mod

# Quick test with fewer samples/epochs
python experiments/run_bukhsh.py --cases WB2 case9mod --quick

# Generate paper figures
python experiments/generate_paper_figures.py
```

## Project Structure

```
PowerSSR-PDNet/
├── src/
│   ├── bukhsh_cases.py      # Bukhsh et al. test cases in pandapower format
│   ├── bukhsh_data.py       # Data generation (LHS sampling + power flow)
│   ├── models.py            # Neural network architectures (Baseline, Physics-NN, SSR-PDNet)
│   ├── trainer.py           # Training loops, metrics, evaluation
│   └── visualization.py     # Security region plots, comparison figures
├── experiments/
│   ├── run_bukhsh.py        # Main experiment script
│   ├── demo.py              # Quick demo on standard IEEE cases
│   └── generate_paper_figures.py  # Final paper figures
├── data/                    # Generated datasets (auto-created)
├── results/                 # Saved model weights and metrics (auto-created)
├── figures/                 # Output figures (auto-created)
└── paper/                   # LaTeX paper draft
```

## Results

### WB2 (2-Bus, Analytical)

| Model         | Accuracy | F1     | Precision | Recall | Specificity |
|---------------|----------|--------|-----------|--------|-------------|
| Baseline      | 0.9967   | 0.9600 | 1.0000    | 0.9231 | 1.0000      |
| Physics-NN    | 0.9967   | 0.9600 | 1.0000    | 0.9231 | 1.0000      |
| **SSR-PDNet** | 0.9900   | 0.8966 | 0.8125    | 1.0000 | 0.9895      |

### WB5 (5-Bus Meshed)

| Model         | Accuracy | F1     | Precision | Recall | Specificity |
|---------------|----------|--------|-----------|--------|-------------|
| Baseline      | 0.9694   | 0.9565 | 0.9285    | 0.9864 | 0.9607      |
| Physics-NN    | 0.9669   | 0.9530 | 0.9235    | 0.9844 | 0.9578      |
| **SSR-PDNet** | **0.9769** | **0.9671** | **0.9391** | **0.9967** | **0.9666** |

### case9mod (Modified IEEE 9-Bus)

| Model         | Accuracy | F1     | Precision | Recall | Specificity |
|---------------|----------|--------|-----------|--------|-------------|
| Baseline      | 0.9720   | 0.9447 | 0.9161    | 0.9752 | 0.9709      |
| Physics-NN    | 0.9714   | 0.9436 | 0.9140    | 0.9752 | 0.9701      |
| **SSR-PDNet** | **0.9860** | **0.9716** | **0.9657** | **0.9777** | **0.9887** |

### case9mod Full-State Surrogate (New)

The repository now includes a full-output surrogate that jointly predicts:
- security feasibility probability;
- OPF state variables: `p1_mw`, `q1_mvar..q3_mvar`, `v1_pu..v9_pu`, `theta1_deg..theta9_deg`.

Run:

```bash
python experiments/train_full_state_surrogate.py --data-dir "D:\安全域\1" --epochs 140 --seed 42
```

Generated artifacts:
- `results/case9mod_fullstate_metrics.json` (classification + state regression metrics)
- `results/case9mod_fullstate_point_comparison.json` (pointwise traditional vs model states)
- `results/case9mod_fullstate_point_comparison.csv` (CSV view of pointwise comparison)

Test-set headline results (case9mod full-state model):
- Classification: `Acc=0.9915, F1=0.9831, Prec=0.9737, Rec=0.9927`
- State MAE groups: `P=0.8366 MW, Q=0.1588 MVAR, V=0.0010 p.u., theta=0.0850 deg`

**Key findings:**
- SSR-PDNet achieves the best overall performance on WB5 and case9mod, while preserving disconnected security-set topology.
- On WB2, SSR-PDNet keeps full recall (1.000) with a more conservative boundary (higher false positives).
- The dual variable $\lambda_v$ stabilizes around 1.15-1.22 in quick runs, indicating active primal-dual constraint regulation.
- The new case9mod full-state surrogate provides internal OPF-state outputs in addition to feasibility labels, with low voltage/angle errors against traditional solutions.

## Mathematical Formulation

### Static Security Region (SSR)

The SSR is defined as the set of load operating points $(P_L, Q_L)$ for which a feasible AC power flow solution exists:

$$\text{SSR} = \{(P_L, Q_L) \in \mathbb{R}^{2n_L} : \exists (V, \theta) \text{ s.t. } f_{PF}(V,\theta,P_L,Q_L) = 0, \; g(V,\theta) \leq 0\}$$

### SSR-PDNet Training Objective

$$\mathcal{L} = \mathcal{L}_{\text{focal}}(\hat{y}, y) + \lambda_{\text{phys}} \cdot \mathcal{L}_{\text{physics}} + \lambda_c \cdot \mathcal{L}_{\text{contrastive}}$$

where:
- $\mathcal{L}_{\text{focal}}$: Focal loss for binary feasibility classification
- $\mathcal{L}_{\text{physics}}$: Voltage constraint violation penalty (with learnable Lagrange multipliers)
- $\mathcal{L}_{\text{contrastive}}$: Contrastive boundary loss for sharp decision boundaries

## References

1. W. A. Bukhsh, A. Grothey, K. McKinnon, P. Trodden, "Local Solutions of Optimal Power Flow Problem," *IEEE Trans. Power Syst.*, vol. 28, no. 4, 2013.

2. H. T. Nguyen, P. L. Donti, "FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees," *NeurIPS*, 2025.

3. E. Liang, M. Chen, S. H. Low, "Low Complexity Homeomorphic Projection to Ensure Neural-Network Solution Feasibility," *ICML*, 2023.

4. M. Kim, H. Kim, "Unsupervised Deep Lagrange Dual With Equation Embedding for AC Optimal Power Flow," *IEEE Trans. Power Syst.*, vol. 40, no. 1, 2025.

5. Z. Hu et al., "Optimal Power Flow Based on Physical-Model-Integrated Neural Network with Worth-Learning Data Generation," *IEEE Trans. Power Syst.*, 2024.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{powerssr2025,
  title={Deep Learning-Based Characterization of Power System Static Security Regions},
  author={Peng Jiao},
  year={2025},
  url={https://github.com/pj0402001/PowerSSR-PDNet}
}
```
