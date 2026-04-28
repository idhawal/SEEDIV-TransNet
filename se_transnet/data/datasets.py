from __future__ import annotations

"""
SEED-IV Dataset utilities -- data loading, splitting, normalization,
and Euclidean Alignment for cross-subject experiments.

CRITICAL UPDATES:
1. Never shuffle windows across sessions before splitting (leakage prevention).
2. Test sets MUST filter out 50%-overlap windows using the generated mask 
   (enforces strict independent testing).
"""

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

# ================================================================
# Constants
# ================================================================

EMOTION_NAMES = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}
NUM_CLASSES = 4
NUM_SUBJECTS = 15
NUM_SESSIONS = 3

SESSION_LABELS = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}

# ================================================================
# Dataset class
# ================================================================

class SeedIVDataset(Dataset):
    """SEED-IV EEG dataset wrapper with per-channel z-score normalization."""

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        domain_labels: np.ndarray | None = None,
        normalise: bool = True,
    ) -> None:
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.domain_labels = (
            domain_labels.astype(np.int64) if domain_labels is not None else None
        )
        self.normalise = normalise

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        eeg = self.data[idx].copy()
        label = int(self.labels[idx])

        if self.normalise:
            mean = eeg.mean(axis=1, keepdims=True)
            std = eeg.std(axis=1, keepdims=True)
            eeg = (eeg - mean) / np.where(std < 1e-8, 1e-8, std)

        if self.domain_labels is None:
            return torch.from_numpy(eeg), label

        domain = int(self.domain_labels[idx])
        return torch.from_numpy(eeg), label, domain

    def class_weights(self) -> torch.Tensor:
        counts = np.array([(self.labels == k).sum() for k in range(NUM_CLASSES)], dtype=np.float32)
        counts = np.where(counts == 0, 1.0, counts)
        weights = 1.0 / counts
        weights = weights / weights.sum() * NUM_CLASSES
        return torch.from_numpy(weights)

    def summary(self, tag: str = '') -> None:
        prefix = f'[{tag}] ' if tag else ''
        parts = [f'{EMOTION_NAMES[k]}: {int((self.labels == k).sum())}' for k in range(NUM_CLASSES)]
        print(f'{prefix}N={len(self)}  shape={self.data.shape}')
        print(f'{prefix}  ' + ' | '.join(parts))

# ================================================================
# Utility Functions
# ================================================================

def euclidean_alignment(X: np.ndarray) -> np.ndarray:
    N, C, T = X.shape
    covs = np.einsum('nct,ndt->ncd', X, X) / T
    R_mean = covs.mean(axis=0)

    eigvals, eigvecs = np.linalg.eigh(R_mean)
    eigvals = np.maximum(eigvals, 1e-10)
    R_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    X_aligned = np.einsum('cd,ndt->nct', R_inv_sqrt, X)
    return X_aligned.astype(np.float32)

def _load_one(dataset_path, sub_id: int, session_id: int) -> tuple:
    base = Path(dataset_path)
    for pattern in [f'sub{sub_id}_session{session_id}', f'sub{sub_id}_sess{session_id}']:
        dp = base / f'{pattern}_data.npy'
        lp = base / f'{pattern}_label.npy'
        mp = base / f'{pattern}_mask.npy'
        if dp.exists():
            data = np.load(str(dp)).astype(np.float32)
            labels = np.load(str(lp)).astype(np.int64)
            # Backwards compatibility: Create a fake mask of 'False' if mask file doesn't exist
            mask = np.load(str(mp)).astype(bool) if mp.exists() else np.zeros(len(labels), dtype=bool)
            return data, labels, mask
    raise FileNotFoundError(f'Missing data for sub{sub_id}_session{session_id} in {base}')

def shuffle_data(data, labels, domains=None, seed=None):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data))
    return (data[idx], labels[idx]) if domains is None else (data[idx], labels[idx], domains[idx])

# ================================================================
# Loaders
# ================================================================

def load_seediv_SD(dataset_path, sub_id: int, config: dict, train_sessions: tuple = (1, 2), test_sessions: tuple = (3,)) -> tuple:
    """Load SEED-IV data with session-based SD split. Test sets exclude overlapping windows."""
    
    def _collect(sessions, is_test=False):
        ds, ls = [], []
        for s in sessions:
            d, l, m = _load_one(dataset_path, sub_id, s)
            if is_test:
                d = d[~m]  # Remove 50% overlapped windows (True in mask)
                l = l[~m]
            ds.append(d)
            ls.append(l)
        return np.concatenate(ds), np.concatenate(ls)

    tr_data, tr_labels = _collect(train_sessions, is_test=False)
    te_data, te_labels = _collect(test_sessions, is_test=True)

    seed = config.get('random_seed', 42)
    tr_data, tr_labels = shuffle_data(tr_data, tr_labels, seed=seed)

    val_frac = config.get('val_split', 0.15)
    n_val = int(len(tr_data) * val_frac)
    
    val_data, val_labels = tr_data[:n_val], tr_labels[:n_val]
    tr_data, tr_labels = tr_data[n_val:], tr_labels[n_val:]

    return tr_data, tr_labels, val_data, val_labels, te_data, te_labels

def load_seediv_LOSO(dataset_path, test_subject: int, config: dict, sessions_to_use: tuple = (1, 2, 3), use_ea: bool = True) -> tuple:
    """Load SEED-IV data with LOSO cross-subject split. Test sets exclude overlapping windows."""
    train_data_list, train_label_list, train_domain_list = [], [], []
    test_data_list, test_label_list = [], []

    source_subjects = [s for s in range(1, NUM_SUBJECTS + 1) if s != test_subject]
    domain_map = {sub_id: i for i, sub_id in enumerate(source_subjects)}

    for sub_id in range(1, NUM_SUBJECTS + 1):
        parts_d, parts_l, parts_m = [], [], []
        for s in sessions_to_use:
            try:
                d, l, m = _load_one(dataset_path, sub_id, s)
                parts_d.append(d)
                parts_l.append(l)
                parts_m.append(m)
            except FileNotFoundError:
                pass
        
        if not parts_d: continue
        
        sub_data = np.concatenate(parts_d, axis=0)
        sub_label = np.concatenate(parts_l, axis=0)
        sub_mask = np.concatenate(parts_m, axis=0)

        if use_ea:
            sub_data = euclidean_alignment(sub_data)

        if sub_id == test_subject:
            # Target testing subject: Remove overlaps
            sub_data = sub_data[~sub_mask]
            sub_label = sub_label[~sub_mask]
            test_data_list.append(sub_data)
            test_label_list.append(sub_label)
        else:
            # Source training subject: Keep everything
            train_data_list.append(sub_data)
            train_label_list.append(sub_label)
            train_domain_list.append(np.full(len(sub_label), domain_map[sub_id], dtype=np.int64))

    train_data = np.concatenate(train_data_list)
    train_labels = np.concatenate(train_label_list)
    train_domains = np.concatenate(train_domain_list)
    test_data = np.concatenate(test_data_list)
    test_labels = np.concatenate(test_label_list)

    seed = config.get('random_seed', 42)
    train_data, train_labels, train_domains = shuffle_data(train_data, train_labels, train_domains, seed=seed)

    val_frac = config.get('val_split', 0.15)
    n_val = int(len(train_data) * val_frac)
    
    val_data, val_labels, val_domains = train_data[:n_val], train_labels[:n_val], train_domains[:n_val]
    train_data, train_labels, train_domains = train_data[n_val:], train_labels[n_val:], train_domains[n_val:]

    return (train_data, train_labels, train_domains, val_data, val_labels, val_domains, test_data, test_labels)