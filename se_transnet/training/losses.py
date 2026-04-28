from __future__ import annotations

"""
Loss functions for SEED-IV emotion recognition.

EmotionDLLoss, GRL, DomainDiscriminator, MMD, CORAL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class EmotionDLLoss(nn.Module):
    """Emotion-aware soft-label cross entropy loss.

    Soft label matrix encodes valence-arousal distances:
      Neutral <-> Sad are close (low valence, low arousal)
      Fear <-> Happy are close (both high arousal)
    """

    def __init__(self, epsilon: float = 0.2, weight: torch.Tensor = None) -> None:
        super().__init__()
        e = epsilon
        dist = torch.tensor(
            [
                [1 - e, e * 0.50, e * 0.25, e * 0.25],
                [e * 0.50, 1 - e, e * 0.25, e * 0.25],
                [e * 0.25, e * 0.25, 1 - e, e * 0.50],
                [e * 0.25, e * 0.25, e * 0.50, 1 - e],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("soft_labels", dist)

        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        soft_targets = self.soft_labels[targets]

        # Native PyTorch soft cross-entropy
        loss = F.cross_entropy(logits, soft_targets, reduction="none")

        if self.weight is not None:
            loss = loss * self.weight[targets]

        return loss.mean()


class GradientReversalLayerFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalLayerFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = alpha


class DomainDiscriminator(nn.Module):
    def __init__(self, in_features: int = 256, n_domains: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_domains),
        )

    def forward(self, x):
        return self.net(x)


def compute_mmd(
    src: torch.Tensor,
    tgt: torch.Tensor,
    bandwidths: list = None,
) -> torch.Tensor:
    """Multi-kernel Maximum Mean Discrepancy loss."""
    if bandwidths is None:
        bandwidths = [0.2, 0.5, 0.9, 1.3]

    def rbf_kernel(x, y, bw):
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist_sq = (diff**2).sum(dim=-1)
        return torch.exp(-dist_sq / (2 * bw))

    mmd = torch.tensor(0.0, device=src.device)
    for bw in bandwidths:
        k_ss = rbf_kernel(src, src, bw).mean()
        k_tt = rbf_kernel(tgt, tgt, bw).mean()
        k_st = rbf_kernel(src, tgt, bw).mean()
        mmd = mmd + (k_ss + k_tt - 2 * k_st)
    return mmd


def compute_coral(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """CORAL loss for matching feature covariances."""
    d = src.size(1)
    ns, nt = src.size(0), tgt.size(0)

    src_c = src - torch.mean(src, dim=0, keepdim=True)
    cov_src = (src_c.t() @ src_c) / (ns - 1 + 1e-6)

    tgt_c = tgt - torch.mean(tgt, dim=0, keepdim=True)
    cov_tgt = (tgt_c.t() @ tgt_c) / (nt - 1 + 1e-6)

    loss = torch.sum((cov_src - cov_tgt) ** 2) / (4 * d * d)
    return loss
