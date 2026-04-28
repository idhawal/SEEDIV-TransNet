from __future__ import annotations

"""
Data augmentation strategies for EEG emotion recognition.

Implements:
  - Signal Segmentation & Recombination (SSR) from EEG-TransNet
  - Left-Right hemisphere swap (leveraging emotion lateralization)
  - Gaussian noise injection
  - Channel dropout (with scale preservation factor)
"""

import random
from typing import Any

import numpy as np
import torch

# ================================================================
# Left-Right electrode pairs for SEED-IV 62-channel layout (0-indexed)
# ================================================================
LR_PAIRS = [
    (0, 2), (3, 4), (5, 13), (6, 12), (7, 11), (8, 10),
    (14, 22), (15, 21), (16, 20), (17, 19), (23, 31),
    (24, 30), (25, 29), (26, 28), (32, 40), (33, 39),
    (34, 38), (35, 37), (41, 49), (42, 48), (43, 47),
    (44, 46), (50, 56), (51, 55), (52, 54), (57, 61),
    (58, 60),
]

def lr_hemisphere_swap(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Swap left-right hemisphere electrode channels."""
    if random.random() >= p:
        return x
    x = x.clone()
    for l_idx, r_idx in LR_PAIRS:
        if x.dim() == 3:
            x[:, l_idx, :], x[:, r_idx, :] = (
                x[:, r_idx, :].clone(),
                x[:, l_idx, :].clone(),
            )
        else:
            x[l_idx, :], x[r_idx, :] = (
                x[r_idx, :].clone(),
                x[l_idx, :].clone(),
            )
    return x

def gaussian_noise(
    x: torch.Tensor,
    sigma_ratio: float = 0.01,
    p: float = 0.3,
) -> torch.Tensor:
    """Add Gaussian noise scaled to signal std."""
    if random.random() >= p:
        return x
    sigma = x.std() * sigma_ratio
    return x + torch.randn_like(x) * sigma

def channel_dropout(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    """Zero out random channels and rescale to preserve signal energy."""
    if p == 0.0: 
        return x
    if p == 1.0: 
        return torch.zeros_like(x)
        
    if x.dim() == 3:
        B, C, T = x.shape
        mask = (torch.rand(B, C, 1) > p).float().to(x.device)
    else:
        C, T = x.shape
        mask = (torch.rand(C, 1) > p).float().to(x.device)
        
    # Scale active nodes to maintain expected sum/variance in next layers
    return (x * mask) / (1.0 - p)


class SSRAugmentor:
    """Signal Segmentation & Recombination (SSR) augmentation."""

    def __init__(
        self,
        num_segs: int = 8,
        num_classes: int = 4,
        batch_size: int = 32,
    ) -> None:
        self.num_segs = num_segs
        self.num_classes = num_classes
        self.aug_per_class = max(1, batch_size // num_classes)

    def __call__(self, data: torch.Tensor, labels: torch.Tensor) -> tuple:
        label_np = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
        data_np = data.cpu().numpy() if torch.is_tensor(data) else np.asarray(data)
        N, C, T = data_np.shape
        seg_size = T // self.num_segs

        aug_data_list = []
        aug_label_list = []

        for cls in range(self.num_classes):
            cls_idx = np.where(label_np == cls)[0]
            if len(cls_idx) <= 1:
                continue

            cls_data = data_np[cls_idx]
            n = len(cls_idx)
            buf = np.zeros((self.aug_per_class, C, T), dtype=np.float32)

            for i in range(self.aug_per_class):
                rand_idx = np.random.randint(0, n, self.num_segs)
                for j in range(self.num_segs):
                    start = j * seg_size
                    end = (j + 1) * seg_size
                    buf[i, :, start:end] = cls_data[rand_idx[j], :, start:end]

            aug_data_list.append(buf)
            aug_label_list.extend([cls] * self.aug_per_class)

        if not aug_data_list:
            return torch.empty(0, C, T), torch.empty(0, dtype=torch.long)

        aug_data = np.concatenate(aug_data_list, axis=0)
        aug_labels = np.array(aug_label_list, dtype=np.int64)
        perm = np.random.permutation(len(aug_data))

        return torch.from_numpy(aug_data[perm]), torch.from_numpy(aug_labels[perm])