"""
Deep learning models for power system static security region characterization.

Three architectures are implemented:
1. BaselineNN: Standard feedforward classifier
2. PhysicsNN: Physics-informed NN with constraint violation penalty
3. SSR_PDNet: Full model — Static Security Region Deep Learning (our proposed method)
   - Uses equation-embedding style hard constraints
   - Lagrange dual training for soft inequality constraints
   - Adversarial boundary sampling for accurate decision boundary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class BaselineNN(nn.Module):
    """
    Baseline: standard feedforward neural network binary classifier.
    Input: normalized load vector [P_load, Q_load]
    Output: probability of being in the static security region
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128, 64]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logits


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in security region boundary regions."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


class BoundaryRegularization(nn.Module):
    """
    Encourages the model to have a sharp, well-defined security boundary.
    Penalizes the norm of the gradient of the predicted probability w.r.t. inputs,
    applied only near the decision boundary (predicted prob ≈ 0.5).
    """

    def __init__(self, margin: float = 0.2, weight: float = 0.01):
        super().__init__()
        self.margin = margin
        self.weight = weight

    def forward(self, logits: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Select points near boundary
        near_boundary = (probs.detach() > 0.5 - self.margin) & (probs.detach() < 0.5 + self.margin)

        if not near_boundary.any():
            return torch.tensor(0.0, device=inputs.device)

        # Compute gradient of logits w.r.t. inputs
        grad = torch.autograd.grad(
            logits[near_boundary].sum(), inputs,
            create_graph=True, retain_graph=True
        )[0][near_boundary]

        # Penalize excessive gradient magnitude at boundary
        return self.weight * grad.norm(dim=-1).mean()


class SSR_PDNet(nn.Module):
    """
    Static Security Region Deep Learning (SSR-PDNet) — our proposed model.

    Key innovations:
    1. Dual-branch architecture: shared feature extractor + separate boundary branch
    2. Lagrange dual training: soft enforcement of physical constraints
    3. Smooth boundary representation with physics-informed regularization
    4. Hard constraint satisfaction via power flow residual minimization

    Architecture:
        Input: x ∈ R^(2n_load)  [normalized P and Q loads]
        Feature Extractor: x → z ∈ R^d (shared representation)
        Classifier Head: z → logit (binary feasibility)
        Physics Head: z → [V_approx, theta_approx] (voltage profile prediction)
    """

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 256,
        classifier_dims: List[int] = None,
        physics_dims: List[int] = None,
        n_bus: int = 30,
        dropout: float = 0.1,
        use_physics_head: bool = True,
    ):
        super().__init__()
        self.use_physics_head = use_physics_head
        self.n_bus = n_bus

        if classifier_dims is None:
            classifier_dims = [512, 512, 256, 128]
        if physics_dims is None:
            physics_dims = [256, 128]

        # Shared feature extractor
        feature_layers = []
        prev_dim = input_dim
        for h in [512, feature_dim]:
            feature_layers.extend([
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        self.feature_extractor = nn.Sequential(*feature_layers)

        # Classifier branch: predicts P(feasible | load)
        cls_layers = []
        prev_dim = feature_dim
        for h in classifier_dims:
            cls_layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.SiLU(),
                nn.Dropout(dropout * 0.5),
            ])
            prev_dim = h

        # Residual connection for last layer
        self.cls_main = nn.Sequential(*cls_layers)
        self.cls_skip = nn.Linear(feature_dim, classifier_dims[-1]) if feature_dim != classifier_dims[-1] else nn.Identity()
        self.cls_out = nn.Linear(classifier_dims[-1], 1)

        # Physics branch: predicts approximate voltage magnitudes
        if use_physics_head:
            phys_layers = []
            prev_dim = feature_dim
            for h in physics_dims:
                phys_layers.extend([
                    nn.Linear(prev_dim, h),
                    nn.SiLU(),
                ])
                prev_dim = h
            phys_layers.append(nn.Linear(prev_dim, n_bus))
            self.physics_head = nn.Sequential(*phys_layers)

        # Learnable Lagrange multipliers (dual variables) for constraint handling
        self.log_lambda_v = nn.Parameter(torch.zeros(1))  # voltage limits
        self.log_lambda_l = nn.Parameter(torch.zeros(1))  # line loading limits

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Returns:
            logit: (B,) feasibility logit
            v_pred: (B, n_bus) predicted voltage magnitudes (or None)
        """
        z = self.feature_extractor(x)

        # Classifier with residual connection
        cls_feat = self.cls_main(z)
        cls_skip = self.cls_skip(z)
        cls_combined = F.silu(cls_feat + cls_skip)
        logit = self.cls_out(cls_combined).squeeze(-1)

        # Physics head
        v_pred = None
        if self.use_physics_head:
            v_raw = self.physics_head(z)
            # Constrain predicted voltages to [0.9, 1.1] p.u.
            v_pred = 0.9 + 0.2 * torch.sigmoid(v_raw)

        return logit, v_pred

    def get_lambda(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current Lagrange multipliers (must be non-negative)."""
        return torch.exp(self.log_lambda_v), torch.exp(self.log_lambda_l)


class PhysicsNN(nn.Module):
    """
    Physics-informed baseline: adds constraint violation penalty to standard NN.
    Uses soft constraint enforcement via penalty terms in loss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        n_bus: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 256, 128]

        # Main classifier
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h

        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, 1)

        # Voltage predictor for physics constraint
        self.v_predictor = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.SiLU(),
            nn.Linear(128, n_bus),
            nn.Sigmoid(),
        )
        self.n_bus = n_bus
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        logit = self.classifier(feat).squeeze(-1)
        v_pred = 0.9 + 0.2 * self.v_predictor(feat)  # [0.9, 1.1] p.u.
        return logit, v_pred


class ContrastiveBoundaryLoss(nn.Module):
    """
    Encourages the model to sharply distinguish the security boundary.
    For feasible/infeasible pairs near the boundary, the model should
    assign significantly different probabilities.

    This acts as a contrastive loss between adjacent feasible-infeasible point pairs.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        feasible_mask = labels > 0.5
        infeasible_mask = ~feasible_mask

        if not (feasible_mask.any() and infeasible_mask.any()):
            return torch.tensor(0.0, device=logits.device)

        # Mean probability for feasible and infeasible points
        p_feas = probs[feasible_mask].mean()
        p_infeas = probs[infeasible_mask].mean()

        # Hinge-style: want p_feas - p_infeas > margin
        loss = F.relu(self.margin - (p_feas - p_infeas))
        return loss
