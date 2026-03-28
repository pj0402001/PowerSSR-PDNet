# Physics-Informed Neural Characterization of Static Security Regions in Generator Power Space Using Multi-Solution Benchmark Cases

> Readable Markdown generated from `paper/main.tex`.
> Mathematical expressions are preserved in LaTeX form (`$...$`, `$$...$$`).

## Abstract

Characterizing the static security region (SSR) in *generator power
space* is fundamental to online security assessment, preventive dispatch, and
operating-margin analysis. The SSR is the set of generator active-power setpoints
for which a voltage- and flow-secure AC solution exists under fixed loading.
For the multi-solution benchmark cases of Bukhsh et al., this set is strongly
nonconvex and may split into disconnected secure components, making dense
nonlinear-feasibility scanning computationally expensive.

We propose **SSR-PDNet**, a physics-informed neural framework that learns the
static security region together with internal operating-state structure. The model combines
(i) a shared encoder operating directly on generator dispatch coordinates,
(ii) a security head for boundary discrimination, and (iii) a physics head that
regularizes the learned representation with voltage-state information through
primal-dual training. This design preserves the correct coordinate-space
interpretation while improving boundary sharpness near disconnected secure
components.

Experiments on WB2, WB5, LMBM3, and case9mod show that SSR-PDNet accurately
reproduces the benchmark static security regions obtained from traditional feasibility
scanning. On the larger multi-component cases, SSR-PDNet attains **F1 = 0.9671**
on WB5 and **F1 = 0.9716** on case9mod while recovering the two- and
three-component secure-set topology, respectively. On WB2 and LMBM3, the paper
further visualizes voltage, reactive-power, redispatch, and bifurcation-related
quantities inside the secure operating set, showing that the framework is not
limited to binary classification but supports physically interpretable security
region analysis.

**Keywords:**
static security region, generator power space,
physics-informed neural network, static security assessment,
Lagrange dual method, multi-solution benchmark cases, disconnected secure components,
power system security assessment.

# Introduction

Power system security assessment requires determining, in near real time,
whether a given operating point will remain within acceptable limits under
normal and contingency conditions. The *static security region* (SSR)
formalizes this as a set in the space of controllable variables (load demands,
generation dispatch) within which all AC power flow constraints — voltage
magnitudes, branch flow limits, reactive power limits — are simultaneously
satisfied [cite].

Traditional SSR computation relies on repeated nonlinear AC power-flow
feasibility solves. Although mature and accurate, these methods scale poorly:
a single large-scale nonlinear solve for a 300-bus system may take seconds on a workstation,
while real-time applications demand sub-millisecond responses across thousands
of candidate operating conditions. Data-driven and machine-learning approaches
have emerged as a promising avenue to reduce this computational burden by
amortizing expensive offline computation into fast online inference
[cite].

## Related Work

**Dispatch surrogate models.** A substantial body of work trains neural
networks to *predict* optimal dispatch decisions given load inputs
[cite]. These models are fast at inference but
produce *point predictions* of the optimal solution rather than
characterizing the security boundary.

**Feasibility-seeking methods.** Nguyen and Donti [cite]
propose FSNet, which appends a differentiable feasibility-seeking correction
step to a standard NN predictor, guaranteeing constraint satisfaction.
Liang et al. [cite] learn a homeomorphic mapping from the
unit ball to the constraint set, enabling strict feasibility via projection.
Kim and Kim [cite] embed power flow equations as hard
constraints via implicit differentiation and enforce inequalities through
Lagrange dual training.

**Security-region characterization.** Jiang and Chiang [cite]
develop a dynamical-systems approach that analytically characterizes AC
security regions as stable equilibrium manifolds of a quotient gradient
system, revealing their potentially disconnected topology. Fan et
al. [cite] quantify the Hausdorff distance between convex
approximation security regions and the true nonconvex static security
region. However, neither approach learns a fast, generalizable model of the
security boundary.

**Multi-solution benchmark cases.** Bukhsh et al. [cite] identify
test cases for which the AC feasibility landscape admits multiple local
solution branches, highlighting the nonconvexity of the static security region. These
cases provide particularly challenging benchmarks for security region methods
because the SSR may be small, irregular, or have non-trivial topology.

## Contributions

This paper makes the following contributions:

- We formulate static security region characterization as binary
    classification in *generator power space* $(P_{G2},\ldots,P_{Gn_g})$
    with loads fixed at nominal values, matching the traditional nonlinear-feasibility scanning
    procedure of Bukhsh et al.\ exactly. This coordinate choice is essential for
    preserving disconnected secure components.

- We propose **SSR-PDNet**, a physics-informed neural architecture with
    a shared encoder, a security discrimination head, and a physics head for
    internal-state regularization. The training objective combines focal loss,
    contrastive boundary sharpening, and primal-dual voltage penalties.

- We enrich the benchmark study beyond classification by visualizing
    secure-point internal states and local point-cloud structure. In WB2 and
    LMBM3, the learned static security region is linked to voltage profiles,
    reactive power, slack redispatch, and dual-solution structure. In WB5 and
    case9mod, boundary-focused zoom figures reveal local point arrangement and
    density heterogeneity that shape boundary formation.

- We conduct systematic experiments on all Bukhsh multi-solution cases.
    SSR-PDNet reproduces both disconnected components of WB5 (F1=0.9671) and all
    three components of case9mod (F1=0.9716), while the redesigned figures also
    show secure-set redispatch and voltage-margin patterns.

- All code, data pipelines (nonlinear-solver and traditional CSV), trained
    models, and visualization tools are released as open source.

The remainder of the paper is organized as follows.
Section [ref] formally defines the SSR and the learning problem.
Section [ref] presents the SSR-PDNet architecture and training
procedure. Section [ref] describes the Bukhsh et al.\ test cases.
Section [ref] presents experimental results including comparison
with traditional feasibility scanning.
Section [ref] discusses limitations and extensions.
Section [ref] concludes.

# Problem Formulation

## AC Power Flow Model

Consider a power network with $n$ buses, $n_g$ generators, $n_\ell$ loads,
and $m$ branches. Let $V_i \in \mathbb{R}_{>0}$ and $\theta_i \in \mathbb{R}$ denote the
voltage magnitude and phase angle at bus $i$. The AC power flow equations at
each bus $i$ are:

$$
P_i^{\mathrm{net}} &= V_i \sum_{j=1}^{n} V_j
    \bigl(G_{ij}\cos\theta_{ij} + B_{ij}\sin\theta_{ij}\bigr), \\
  Q_i^{\mathrm{net}} &= V_i \sum_{j=1}^{n} V_j
    \bigl(G_{ij}\sin\theta_{ij} - B_{ij}\cos\theta_{ij}\bigr),
$$

where $\theta_{ij} = \theta_i - \theta_j$, and $G_{ij} + jB_{ij}$ is the
$(i,j)$ entry of the bus admittance matrix $\mathbf{Y}_{\mathrm{bus}}$. Net
power injections are $P_i^{\mathrm{net}} = P_i^G - P_i^L$ and
$Q_i^{\mathrm{net}} = Q_i^G - Q_i^L$.

## Static Security Region

**Definition (Static Security Region).**

  Let $\mathbf{u} = (P_{G2},\ldots,P_{Gn_g})^\top \in \mathbb{R}^{n_g-1}$ denote the
  vector of controllable generator active-power setpoints (excluding the slack
  bus generator whose output $P_{G1}$ adjusts automatically to maintain power
  balance). With loads *fixed* at nominal values
  $\mathbf{P}^L_0, \mathbf{Q}^L_0$, the static security region is defined as:

$$
\mathcal{S} = \bigl\{\mathbf{u} \in \mathbb{R}^{n_g-1} :
      \exists\,(V, \theta, Q^G) \text{ satisfying }
      \eqref{eq:pf_p}\text{--}\eqref{eq:pf_q}\text{ and }
      \eqref{eq:volt_lim}\text{--}\eqref{eq:line_lim} \bigr\},
$$

  where the operational constraints include:

$$
V_i^{\min} &\le V_i \le V_i^{\max}, \quad \forall i, \\
    P_g^{\min} &\le P_g^G \le P_g^{\max}, \quad \forall g, \\
    Q_g^{\min} &\le Q_g^G \le Q_g^{\max}, \quad \forall g, \\
    S_k &:= \sqrt{P_k^2 + Q_k^2} \le S_k^{\max}, \quad \forall k.
$$

**Remark.**

  **Critical distinction:** The static security region $\mathcal{S}$ is defined
  in *generator power space* $\mathbf{u} = (P_{G2},\ldots,P_{Gn_g})$,
  with loads treated as *fixed parameters*. This is the standard
  formulation used in traditional nonlinear-feasibility scanning [cite] and
  quotient gradient system theory [cite]. For a given
  $\mathbf{u} \in \mathcal{S}$, the slack-bus generator output
  $P_{G1} = \sum_i P_i^L + P_{\mathrm{loss}} - \sum_{g \ge 2} P_{Gg}$
  automatically. The reactive dispatch $Q^G$ and voltage profile $(V,\theta)$
  are the remaining variables solved by the AC power flow equations.

**Theorem (Topology of $\mathcal{S}$ [cite]).**

  Under generic conditions, each connected component of $\mathcal{S}$ is a
  smooth submanifold of $\mathbb{R}^{n_g-1}$, corresponding to a *regular stable
  equilibrium manifold* (SEM) of the quotient gradient system (QGS)
  $\dot{\mathbf{u}} = -D H(\mathbf{u})^\top H(\mathbf{u})$, where
  $H(\mathbf{u}) = 0$ encodes the AC power flow equations with $\mathbf{u}$ fixed.
  Disconnected components arise at *pseudo-pitchfork bifurcation points*
  where two SEMs merge or split.

**Remark.**

  $\mathcal{S}$ is generally nonconvex and may be *disconnected* [cite].
  For the Bukhsh et al. benchmark cases:
  WB5 has **2 disconnected components** in $(P_{G1}, P_{G5})$ space;
  case9mod has **3 disconnected components** in $(P_{G2}, P_{G3})$ space;
  LMBM3 has **2 components** for load factor $\lambda \approx 1.14$--$1.5$.
  The Hausdorff distance between the SDP convex relaxation and $\mathcal{S}$
  grows rapidly near the saddle-node bifurcation point [cite].

## Traditional Nonlinear-Feasibility Scanning (Ground Truth)

The traditional method for computing $\mathcal{S}$ fixes $\mathbf{u}$ to a
grid point and solves:

$$
\min_{V, \theta, Q^G}\; 0 \quad \text{s.t.} \quad
  \eqref{eq:pf_p}\text{--}\eqref{eq:pf_q},\;
  \eqref{eq:volt_lim}\text{--}\eqref{eq:line_lim},\;
  P_{Gg} = u_g\; \forall g \ge 2.
$$

The point $\mathbf{u}$ is declared secure if the nonlinear feasibility solver
(tolerance $10^{-8}$, max 3000 iterations) converges to a feasible AC state [cite].
To overcome non-convexity and find all secure components, a multi-start
strategy is used: (i) *flat start* $V=1$, $\theta=0$; (ii) *outer start*
with pre-specified angle estimates from the MATPOWER solution. Warm-starting
propagates solutions along each scan direction.

## Learning Problem

We formulate SSR characterization as supervised binary classification in
*generator power space*. Given a labeled dataset
$\mathcal{D} = \{(\mathbf{u}^{(i)}, y^{(i)})\}_{i=1}^{N}$ where
$\mathbf{u}^{(i)} = (P_{G2}^{(i)},\ldots,P_{Gn_g}^{(i)})^\top \in \mathbb{R}^{n_g-1}$
and $y^{(i)} \in \{0, 1\}$ is the traditional-solver security label
($1$ = secure, $0$ = insecure), we seek a function
$f_\phi : \mathbb{R}^{n_g-1} \to [0,1]$ predicting $\Pr(y=1 \mid \mathbf{u})$.

Training data for WB5 and case9mod is taken directly from pre-computed traditional
scanning results (32,068 secure points for WB5; 2,735 for case9mod), augmented
with uniformly sampled background insecure points (2:1 ratio). This ensures
the training distribution matches the ground-truth distribution from the
traditional solver. Labels satisfy: $y^{(i)} = 1 \iff \mathbf{u}^{(i)} \in \mathcal{S}$.

# SSR-PDNet: Proposed Method

## Architecture Overview

SSR-PDNet consists of three components (see Fig. [ref] and the description below):

- **Shared Feature Extractor** $\phi_z : \mathbb{R}^{n_u} \to \mathbb{R}^d$:
    maps normalized generator-dispatch coordinates to a latent representation
    $\mathbf{z} = \phi_z(\mathbf{x})$.

- **Security Head** $\phi_c : \mathbb{R}^d \to \mathbb{R}$: produces a security
    logit $\hat{\ell} = \phi_c(\mathbf{z})$.

- **Physics Head** $\phi_v : \mathbb{R}^d \to \mathbb{R}^n$: predicts bus voltage
    magnitudes $\hat{V} = \phi_v(\mathbf{z})$, constrained to $[0.9, 1.1]$ p.u.\ via
    a sigmoid output layer.

The shared feature extractor uses two fully-connected layers with LayerNorm and
SiLU activations. The classifier branch has four hidden layers with
BatchNorm, SiLU, and a residual skip connection:

$$
\phi_c(\mathbf{z}) = W_{\mathrm{out}} \cdot \mathrm{SiLU}\bigl(
    \phi_c^{\mathrm{main}}(\mathbf{z}) + W_{\mathrm{skip}} \mathbf{z} \bigr).
$$

> [Figure omitted; see the figures directory.]

## Training Objective

The total training loss is:

$$
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{focal}}
    + \lambda_{\mathrm{phys}} \cdot \mathcal{L}_{\mathrm{physics}}
    + \lambda_c \cdot \mathcal{L}_{\mathrm{contrastive}},
$$

where $\lambda_{\mathrm{phys}} = 0.1$ and $\lambda_c = 0.05$ are
hyperparameters.

### Focal Loss

To handle class imbalance — the SSR may occupy only a limited fraction of the
sampled space in boundary-focused datasets — we use the focal loss [cite]:

$$
\mathcal{L}_{\mathrm{focal}} = -\frac{1}{N} \sum_{i=1}^{N}
    \alpha_t (1 - p_t)^\gamma \log p_t,
$$

with $\alpha = 0.75$ and $\gamma = 2.0$, where $p_t$ is the model's predicted
probability for the true class.

### Physics Constraint Loss with Lagrange Dual Training

For secure samples ($y^{(i)}=1$), the predicted voltage profile $\hat{V}$
should satisfy the voltage limits. We enforce this through a Lagrange penalty:

$$
\mathcal{L}_{\mathrm{physics}} = \lambda_V \cdot
    \frac{1}{|\mathcal{S}_\mathcal{D}|} \sum_{i \in \mathcal{S}_\mathcal{D}}
    \sum_{k=1}^{n} \bigl[\mathrm{ReLU}(V_{\min} - \hat{V}_k^{(i)})
      + \mathrm{ReLU}(\hat{V}_k^{(i)} - V_{\max})\bigr],
$$

where $\mathcal{S}_\mathcal{D} = \{i : y^{(i)} = 1\}$ and $\lambda_V > 0$ is a learnable
Lagrange multiplier (dual variable). The dual variable is updated by gradient
*ascent* on $\mathcal{L}_{\mathrm{total}}$, while the primal network weights are
updated by gradient *descent*, following the primal-dual framework
of [cite]:

$$
\phi &\leftarrow \phi - \eta_\phi \nabla_\phi \mathcal{L}_{\mathrm{total}},
    \\
  \log\lambda_V &\leftarrow \log\lambda_V + \eta_\lambda
    \nabla_{\log\lambda_V} \mathcal{L}_{\mathrm{total}}.
$$

### Contrastive Boundary Loss

To encourage a sharp, well-defined security boundary, we apply a margin-based
contrastive loss between the average predicted probability of secure and
insecure samples in each mini-batch:

$$
\mathcal{L}_{\mathrm{contrastive}} = \mathrm{ReLU}\!\left(
    m - \bigl(\bar{p}_{\mathcal{S}} - \bar{p}_{\mathcal{I}}\bigr)
  \right),
$$

where $\bar{p}_{\mathcal{S}}$ and $\bar{p}_{\mathcal{I}}$ are the mean
predicted probabilities over secure and insecure samples, respectively, and
$m = 0.5$ is the target margin.

## Training Procedure

> [Algorithm omitted in readable markdown.]

The primal optimizer is AdamW with cosine annealing learning rate schedule
($\eta_\phi^0 = 10^{-3}$, $\eta_\phi^{\min} = 10^{-5}$). The dual optimizer
is Adam ($\eta_\lambda = 10^{-2}$). Class-balanced mini-batch sampling via
weighted random sampler ensures each class appears equally in expectation.

# Bukhsh et al.\ Test Cases

We use the test cases from Bukhsh et al. [cite], originally
provided in MATPOWER format. We implement them in both pandapower (Python) and
a direct Pyomo-based nonlinear-feasibility framework for reproducibility. Table [ref] summarizes
the cases. **Critically**, all experiments are conducted in
*generator power space* $\mathbf{u} = (P_{G2},\ldots,P_{Gn_g})$ with loads
fixed at nominal values, matching the traditional nonlinear-feasibility scanning approach exactly.

> [Table omitted in readable markdown.]

## WB2: 2-Bus Analytical Case

WB2 has a single slack generator at bus 1 ($V_1 = 0.964$ p.u.) and a load at bus 2
with $P_d = 350$ MW and $Q_d = -350$ MVAR (capacitive). Since there is only one
generator, the SSR in generator power space is degenerate (only the slack bus
generator exists). We instead characterize the secure operating set in the 2D
$(P_d, Q_d)$ space, which directly corresponds to generator output
$P_{G1} = P_d + P_{\mathrm{loss}}$.

**Proposition (WB2 Dual Solutions [cite]).**

  For the nominal load $(P_d, Q_d) = (350, -350)$ MW/MVAR, the AC power flow
  equations admit exactly two solutions with bus-2 voltages $V_2 \approx 0.922$ p.u.
  and $V_2 \approx 1.095$ p.u., both violating the voltage limit
  $[0.95, 1.05]$ p.u. The secure operating set is a narrow strip in the
  $(P_d, Q_d)$ plane where a voltage-secure solution exists, with
  secure-set area fraction $\approx 3.2\%$ (see Fig. [ref]).

> [Figure omitted; see the figures directory.]

## WB5: 5-Bus Meshed Case

WB5 has two generators at bus 1 (slack, G1) and bus 5 (PV, G5), with loads
fixed at $[P_{d2}, P_{d3}, P_{d4}] = [130, 130, 65]$ MW and
$[Q_{d2}, Q_{d3}, Q_{d4}] = [20, 20, 10]$ MVAR. The voltage limits are
$[0.87, 1.13]$ p.u. and the line limits are $S_k^{\max} = 2500$ MVA.

**Traditional feasibility scanning** (our ground truth) fixes both $P_{G1}$ and
$P_{G5}$ and scans $(P_{G5} \in [0, 400]$ MW, $P_{G1} \in [0, 700]$ MW) with
3-stage multi-start (coarse $\to$ fine $\to$ gap-fill), yielding **32,068
secure points** forming **2 disconnected components** in $(P_{G1}, P_{G5})$
space (see Fig. [ref]).

The disconnected structure arises from the non-convexity of the line flow
constraints: for certain $(P_{G1}, P_{G5})$ combinations, no voltage profile
$(V, \theta)$ satisfies the AC power flow equations simultaneously with all
line thermal limits. This is a *pseudo-pitchfork bifurcation* in the
quotient gradient system [cite].

> [Figure omitted; see the figures directory.]

## case9mod: Modified IEEE 9-Bus

case9mod modifies the standard IEEE 9-bus case with: (i) $Q_{\min} = -5$ MVAR
(tightened from $-300$) for all generators; (ii) loads at 60\% of standard
values ($[P_{d5}, P_{d7}, P_{d9}] = [54, 60, 75]$ MW). Generators at buses 2 and 3
are the controllable variables; bus 1 is the slack.

**Traditional feasibility scanning** fixes $(P_{G2}, P_{G3})$ and solves
security feasibility, yielding **2,735 secure points** in the
$(P_{G2}, P_{G3}) \in [10, 300] \times [10, 270]$ MW space forming
**3 disconnected components**. The tight $Q_{\min}$ constraint is the
primary source of insecurity and component separation.

> [Figure omitted; see the figures directory.]

\FloatBarrier

# Experiments

## Experimental Setup

**Data generation.** Training data comes from pre-computed traditional scanning
results (the same ground truth used for visualization). Insecure background
points are sampled uniformly in the generator power space at a 2:1
(insecure:secure) ratio. Table [ref] summarizes the datasets.

> [Table omitted in readable markdown.]

**Baselines.**

- **Baseline NN**: Standard feedforward classifier with focal loss,
    hidden layers $[256, 256, 128, 64]$, SiLU activations, BatchNorm, dropout 0.1.

- **Physics-NN**: Adds a voltage-prediction branch and constraint
    violation penalty term to the Baseline NN, but without learnable dual
    variables or contrastive loss.

**Train/val/test split.** 70\%/15\%/15\% split; class-balanced training
with weighted random sampler. Metrics evaluated on the held-out test set.

**Hyperparameters.** See Table [ref]. Early stopping with
patience 40 on validation F1 score.

> [Table omitted in readable markdown.]

**Hardware.** All experiments run on CPU (Intel, Windows 10) with
PyTorch 2.8. Inference time is sub-millisecond per sample on all cases.

## Security Region Visualization

Fig. [ref] and Fig. [ref] show the WB5 static security region in *generator
power space* $(P_{G1}, P_{G5})$. The traditional ground-truth set (Fig. [ref]) reveals
**2 disconnected secure components** — a key topological property of the
true SSR arising from the nonconvexity of AC power flow constraints at certain
$(P_{G1}, P_{G5})$ combinations. The SSR-PDNet predicted probability map (right)
correctly reproduces both components, with two high-confidence green regions
separated by an insecure gap. This result would be impossible in the
*wrong* load-space formulation, which cannot exhibit this disconnected structure.

> [Figure omitted; see the figures directory.]

> [Figure omitted; see the figures directory.]

Fig. [ref] and Fig. [ref] show the case9mod static security region in
$(P_{G2}, P_{G3})$ generator power space. The traditional ground-truth set (Fig. [ref])
reveals **3 disconnected secure components**, caused by the tight
$Q_{\min}=-5$ MVAR constraint creating insecure bands between different
reactive-power-limited operating regimes. SSR-PDNet reproduces all three components
with F1 = 0.9716 and outperforms both Baseline and Physics-NN under this split.

> [Figure omitted; see the figures directory.]

> [Figure omitted; see the figures directory.]

To make the formation mechanism of these regions visually clear at the local scale,
Fig. [ref] and Fig. [ref] provide boundary-focused
zoom-in views. Each zoom window is centered on a boundary-dense subregion and
reports: (a) secure-point scatter layout, (b) hexbin density map with local
point-density statistics, and (c) SSR-PDNet score field with the true boundary
overlay. These panels show that the region boundary is formed by highly
nonuniform point arrangements, with dense ridges and sparse transition bands,
rather than by a uniformly filled cloud.

\FloatBarrier

## Quantitative Results

Table [ref] presents classification metrics on the held-out test sets,
all computed in the correct *generator power space* against traditional ground truth.

To verify whether the framework can approximate not only security labels but
also the *underlying OPF state*, we additionally trained a full-output
surrogate on case9mod that jointly predicts: (i) security feasibility and
(ii) internal state variables
$\{P_{G1},Q_{G1:3},V_{1:9},\theta_{1:9}\}$. This model uses the same
traditional samples and reports pointwise state errors only on feasible
operating points where traditional state labels exist.

> [Table omitted in readable markdown.]

**WB2.** On the highly imbalanced WB2 split, SSR-PDNet keeps full recall
(1.000) with F1 = 0.8966. Compared with Baseline and Physics-NN (F1 = 0.9600),
it is more conservative in security labeling (higher false-positive rate), which
is still useful when missed secure points are prioritized over stricter precision.

**WB5.** SSR-PDNet achieves F1 = 0.9671 and near-perfect recall (0.9967),
correctly identifying essentially all secure $(P_{G1}, P_{G5})$ operating
points (ROC/PR curves in Fig. [ref]). The high recall is safety-critical: missing a secure point means
unnecessarily restricting dispatch. SSR-PDNet correctly reproduces *both
disconnected secure components* of the WB5 static security region.

> [Figure omitted; see the figures directory.]

**case9mod.** SSR-PDNet achieves F1 = 0.9716, outperforming Baseline NN
(0.9447) and Physics-NN (0.9436) under this split.
(ROC/PR curves in Fig. [ref]).
More importantly, SSR-PDNet correctly reproduces all three disconnected secure
components in $(P_{G2}, P_{G3})$ space — a structural property that cannot
be captured by models operating in load space.

> [Figure omitted; see the figures directory.]

**LMBM3.** Both load-factor settings ($\lambda=1.490$ and $\lambda=1.500$)
yield perfect F1 = 1.000 and Recall = 1.000 for all models, including SSR-PDNet.
This reflects a well-separated secure-set structure in $(P_{G1}, P_{G2})$ space
at these near-collapse operating points: once the traditional labels provide accurate
labels, the classification task is straightforward. The interesting behavior
is the topological transition (bifurcation) between $\lambda$ values, visualized
in Fig. [ref].

> [Figure omitted; see the figures directory.]

**Grid and boundary comparison.** Table [ref] reports
grid-level agreement and boundary-focused scores relative to the traditional
ground truth.

> [Table omitted in readable markdown.]

**Full-state surrogate accuracy (case9mod).**
Table [ref] reports the multitask surrogate results. In
addition to high security-classification accuracy, the model achieves low
state-variable errors: voltage MAE is around $10^{-3}$ p.u., angle MAE is
around $10^{-1}$ degree, and reactive-power MAE is around $10^{-1}$ MVAR.
These results indicate that the learned mapping captures not only the boundary
topology of the static security region, but also the physical operating-state
structure behind each secure point.

> [Table omitted in readable markdown.]

**Pointwise state comparison.**
Representative points are listed in the released comparison file
`results/case9mod\_fullstate\_point\_comparison.csv`, where each row
contains input dispatch, traditional feasibility label, surrogate probability,
predicted states, traditional states, and grouped MAE statistics.

## Training Dynamics

SSR-PDNet converges to strong validation F1 within $\sim 40$ epochs on case9mod
and $\sim 80$ epochs on WB5 (Figs. [ref] and
[ref]). The Lagrange multiplier $\lambda_V$ stabilizes
around $1.15$--$1.22$ after epoch 20, indicating a stable primal-dual
equilibrium under voltage-margin penalties. Early stopping patience of
35--40 epochs helps prevent overfitting to the background insecure class.

> [Figure omitted; see the figures directory.]

> [Figure omitted; see the figures directory.]

\FloatBarrier

# Discussion

## Importance of Correct Coordinate Space

The most critical finding of this work is that **the choice of coordinate
space fundamentally determines whether the SSR can be correctly characterized**.
Previous approaches operating in load space (varying $P_L, Q_L$) with fixed
generators are solving the wrong problem: the SSR as defined by Bukhsh et al.\
and Jiang \& Chiang is in *generator power space*, not load space.

In load space, the security boundary reflects which load conditions admit
a secure dispatch — but loads are typically known and fixed in operation;
the controllable variables are generator outputs. In generator power space
(with loads fixed at nominal), the SSR directly answers: ``for this dispatch
plan $(P_{G2}, P_{G3})$, does a secure AC operating point exist?''
This is the operationally relevant question for security assessment.

The disconnected component structure (2 components for WB5, 3 for case9mod)
is a topological property of the SSR in generator power space that stems from
the multiple local solutions of the AC power flow equations [cite]
and the quotient gradient system bifurcation theory [cite].
A model trained in load space cannot reproduce this structure.

## Why SSR-PDNet Outperforms Baselines

The key advantages of SSR-PDNet over the Baseline NN are:
(1) the *contrastive boundary loss*, which explicitly widens the margin
between secure and insecure predictions, particularly effective near the
thin insecure corridors separating the 3 components in case9mod;
(2) the *residual classifier* with skip connection, which preserves
gradient information through deep layers; and
(3) the *Lagrange dual training*, which promotes conservative security
predictions by penalizing predicted-secure points whose voltage profiles
violate the physics constraints.

## Limitations and Future Work

**Background insecure data quality.** The background insecure points
are sampled uniformly, which may include points far from the boundary and
create an imbalanced difficulty distribution. Worth-learning adversarial
sampling [cite] targeting the boundary region would improve
boundary F1 further.

**LMBM3.** The LMBM3 case exhibits a narrow secure band that requires
dense feasibility scanning with load-factor parametrization ($\lambda \in [1.0, 1.5]$)
to capture the bifurcation behavior. Fig. [ref] visualizes the
pre-computed traditional results at $\lambda=1.490$ (pre-bifurcation narrow
secure band) and $\lambda=1.500$ (voltage collapse onset). Under the current
dataset construction, DL classification is nearly separable and all compared
models achieve F1 = 1.000 on both reported load factors.

**Scalability.** For larger systems (100+ buses), dense nonlinear grid scanning
approach becomes intractable. Adaptive boundary sampling strategies and GNN-based
architectures exploiting network topology [cite] are promising
directions.

**Dynamic security.** Extension from static security regions to transient
stability and voltage-stability regions is an important direction for future work.

## Comparison with Related Work

Unlike dispatch surrogate methods [cite] that predict *optimal*
dispatch, SSR-PDNet predicts *security-region membership* in generator power space —
the safety-critical question of whether a candidate dispatch admits any
AC-secure solution. Unlike FSNet [cite] or homeomorphic
projection [cite], which enforce feasibility of a
*single* predicted solution, SSR-PDNet characterizes the entire SSR boundary
as a global structure.

The closest prior work is Jiang and Chiang [cite], which
analytically characterizes the same SSR using quotient gradient system theory.
Our approach learns the same boundary from traditional labeled data, trading analytical rigor
for scalability and the ability to handle arbitrary network topology without
closed-form power flow inversions.

# Conclusion

We presented SSR-PDNet, a physics-informed neural framework for characterizing the
static security region in *generator power space* — the correct coordinate
space for power system security assessment. By working in generator power space
with loads fixed at nominal, SSR-PDNet reproduces the disconnected component
structure (2 components for WB5, 3 for case9mod) that is the defining topological
feature of the static security region in the Bukhsh et al.\ benchmark cases. SSR-PDNet
achieves F1 = 0.9671 on WB5 and F1 = 0.9716 on case9mod, with near-perfect
recall (0.9967) on WB5 and full recall (1.000) on WB2, critical for operational safety.

Beyond boundary characterization, the added full-state surrogate study on
case9mod demonstrates that a neural surrogate can jointly recover security
labels and internal OPF state variables with high fidelity relative to the
traditional solver, supporting a practical path toward replacing repetitive
pointwise nonlinear solves in online security-assessment workflows.

Future work will address: (i) scalability to large systems via graph neural
network encoders that exploit network topology; (ii) extension to $N{-}1$
contingency security regions; (iii) integration with online security assessment
workflows; and (iv) incorporation of the worth-learning data generation strategy
to efficiently sample near the security boundary.

\bibliographystyle{IEEEtran}
\bibliography{references}
