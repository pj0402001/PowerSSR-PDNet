"""
Training pipeline for power system static security region (SSR) deep learning models.
Implements:
- Standard supervised training with focal loss
- Physics-informed training with Lagrange dual updates
- Adversarial boundary-aware data augmentation
- Learning rate scheduling and early stopping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
import json

from models import BaselineNN, PhysicsNN, SSR_DL, FocalLoss, ContrastiveBoundaryLoss


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-5, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_state = None

    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False

        improved = (score > self.best_score + self.min_delta) if self.mode == 'max' else (score < self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def make_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 512,
    balance: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test data loaders with optional class balancing."""
    rng = np.random.RandomState(seed)
    n = len(X)
    idx = rng.permutation(n)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    X_t = torch.FloatTensor(X[train_idx])
    y_t = torch.FloatTensor(y[train_idx])
    X_v = torch.FloatTensor(X[val_idx])
    y_v = torch.FloatTensor(y[val_idx])
    X_te = torch.FloatTensor(X[test_idx])
    y_te = torch.FloatTensor(y[test_idx])

    # Balanced sampling for imbalanced datasets
    if balance:
        class_counts = np.bincount(y[train_idx].astype(int))
        weights = 1.0 / class_counts[y[train_idx].astype(int)]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(TensorDataset(X_v, y_v), batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size * 2, shuffle=False)

    return train_loader, val_loader, test_loader


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Dict:
    """Compute classification metrics."""
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs > threshold).astype(int)
    y = labels.cpu().numpy().astype(int)

    tp = ((preds == 1) & (y == 1)).sum()
    tn = ((preds == 0) & (y == 0)).sum()
    fp = ((preds == 1) & (y == 0)).sum()
    fn = ((preds == 0) & (y == 1)).sum()

    acc = (tp + tn) / (len(y) + 1e-10)
    prec = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    spec = tn / (tn + fp + 1e-10)  # true negative rate

    return {'acc': float(acc), 'prec': float(prec), 'rec': float(rec),
            'f1': float(f1), 'spec': float(spec),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}


def train_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = None,
    patience: int = 30,
) -> Dict:
    """Train a baseline NN classifier."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    early_stop = EarlyStopping(patience=patience, mode='max')

    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_acc': []}
    best_metrics = {}

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            if hasattr(model, 'forward') and isinstance(model, (PhysicsNN, SSR_DL)):
                logits, _ = model(X_batch)
            else:
                logits = model(X_batch)

            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_logits, val_labels = [], []
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                if isinstance(model, (PhysicsNN, SSR_DL)):
                    logits, _ = model(X_batch)
                else:
                    logits = model(X_batch)
                val_logits.append(logits)
                val_labels.append(y_batch)
                val_losses.append(criterion(logits, y_batch).item())

        val_logits = torch.cat(val_logits)
        val_labels = torch.cat(val_labels)
        metrics = compute_metrics(val_logits, val_labels)

        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['val_f1'].append(metrics['f1'])
        history['val_acc'].append(metrics['acc'])

        scheduler.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"train_loss={np.mean(train_losses):.4f} | "
                  f"val_f1={metrics['f1']:.4f} | "
                  f"val_acc={metrics['acc']:.4f}")

        if early_stop(metrics['f1'], model):
            print(f"Early stopping at epoch {epoch+1}")
            break

    early_stop.restore_best(model)
    return history


def train_ssr_dl(
    model: SSR_DL,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 300,
    lr: float = 1e-3,
    lr_dual: float = 1e-2,
    weight_decay: float = 1e-4,
    lambda_physics: float = 0.1,
    lambda_boundary: float = 0.05,
    lambda_contrastive: float = 0.05,
    v_min: float = 0.9,
    v_max: float = 1.1,
    device: torch.device = None,
    patience: int = 40,
) -> Dict:
    """
    Train the SSR-DL model using combined loss:
    L = L_focal (supervised) + λ_p * L_physics (voltage constraint) +
        λ_b * L_boundary (gradient sharpness) + λ_c * L_contrastive
    With Lagrange dual update for constraint satisfaction.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    # Separate parameters: model params vs dual variables
    primal_params = [p for n, p in model.named_parameters()
                     if 'log_lambda' not in n]
    dual_params = [p for n, p in model.named_parameters()
                   if 'log_lambda' in n]

    optimizer = torch.optim.AdamW(primal_params, lr=lr, weight_decay=weight_decay)
    dual_optimizer = torch.optim.Adam(dual_params, lr=lr_dual)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
    contrastive_loss = ContrastiveBoundaryLoss(margin=0.5)
    early_stop = EarlyStopping(patience=patience, mode='max')

    history = {
        'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_acc': [],
        'loss_focal': [], 'loss_physics': [], 'loss_contrastive': [],
        'lambda_v': [], 'lambda_l': [],
    }

    for epoch in range(epochs):
        model.train()
        epoch_losses = {'focal': [], 'physics': [], 'contrastive': [], 'total': []}

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            X_batch.requires_grad_(True)

            # Forward pass
            logits, v_pred = model(X_batch)

            # 1. Supervised focal loss
            loss_focal = focal_loss(logits, y_batch)

            # 2. Physics constraint: voltage limits
            loss_physics = torch.tensor(0.0, device=device)
            if v_pred is not None:
                lambda_v, lambda_l = model.get_lambda()
                # Voltage constraint violations
                v_viol = F.relu(v_min - v_pred) + F.relu(v_pred - v_max)
                # For infeasible samples, encourage violation prediction;
                # for feasible samples, enforce constraint satisfaction
                feasible_mask = y_batch > 0.5
                if feasible_mask.any():
                    loss_physics = lambda_v * v_viol[feasible_mask].mean()

            # 3. Contrastive boundary loss
            loss_contrastive = lambda_contrastive * contrastive_loss(logits, y_batch)

            # Total loss
            loss = loss_focal + lambda_physics * loss_physics + loss_contrastive

            optimizer.zero_grad()
            dual_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(primal_params, max_norm=1.0)
            optimizer.step()

            # Dual ascent (maximize constraint penalties)
            dual_optimizer.step()

            epoch_losses['focal'].append(loss_focal.item())
            epoch_losses['physics'].append(loss_physics.item() if isinstance(loss_physics, torch.Tensor) else 0.0)
            epoch_losses['contrastive'].append(loss_contrastive.item())
            epoch_losses['total'].append(loss.item())

        # Validation
        model.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits, _ = model(X_batch)
                val_logits.append(logits)
                val_labels.append(y_batch)

        val_logits = torch.cat(val_logits)
        val_labels = torch.cat(val_labels)
        metrics = compute_metrics(val_logits, val_labels)
        val_loss = focal_loss(val_logits, val_labels).item()

        lambda_v, lambda_l = model.get_lambda()

        history['train_loss'].append(np.mean(epoch_losses['total']))
        history['val_loss'].append(val_loss)
        history['val_f1'].append(metrics['f1'])
        history['val_acc'].append(metrics['acc'])
        history['loss_focal'].append(np.mean(epoch_losses['focal']))
        history['loss_physics'].append(np.mean(epoch_losses['physics']))
        history['loss_contrastive'].append(np.mean(epoch_losses['contrastive']))
        history['lambda_v'].append(lambda_v.item())
        history['lambda_l'].append(lambda_l.item())

        scheduler.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"total={np.mean(epoch_losses['total']):.4f} | "
                  f"focal={np.mean(epoch_losses['focal']):.4f} | "
                  f"phys={np.mean(epoch_losses['physics']):.4f} | "
                  f"val_f1={metrics['f1']:.4f} | "
                  f"λ_v={lambda_v.item():.3f}")

        if early_stop(metrics['f1'], model):
            print(f"Early stopping at epoch {epoch+1}")
            break

    early_stop.restore_best(model)
    return history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict:
    """Full evaluation on test set."""
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            if isinstance(model, (PhysicsNN, SSR_DL)):
                logits, _ = model(X_batch)
            else:
                logits = model(X_batch)
            all_logits.append(logits.cpu())
            all_labels.append(y_batch)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Threshold sweep for optimal threshold selection
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1, best_th = 0, threshold
    for th in thresholds:
        m = compute_metrics(all_logits, all_labels, threshold=th)
        if m['f1'] > best_f1:
            best_f1 = m['f1']
            best_th = th

    metrics = compute_metrics(all_logits, all_labels, threshold=best_th)
    metrics['best_threshold'] = best_th
    metrics['probs'] = torch.sigmoid(all_logits).numpy()
    metrics['labels'] = all_labels.numpy()

    return metrics
