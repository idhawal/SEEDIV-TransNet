from __future__ import annotations

"""
Subject-Dependent training loop for SE-TransNet.
AMP, gradient clipping, SSR+LR+noise augmentation, CosineAnnealingLR, early stopping.
"""

import copy
import os
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from se_transnet.data.augmentation import (
    SSRAugmentor,
    gaussian_noise,
    lr_hemisphere_swap,
)
from se_transnet.data.datasets import SeedIVDataset


def _safe_kappa(y_true, y_pred):
    """Cohen's kappa with NaN guard for degenerate predictions.

    cohen_kappa_score raises RuntimeWarning and returns NaN when all
    predictions (or all true labels) belong to a single class because the
    expected agreement denominator becomes 0.  We return 0.0 in that case
    rather than propagating NaN into summary statistics.
    """
    import math

    try:
        k = cohen_kappa_score(y_true, y_pred)
        return 0.0 if (isinstance(k, float) and math.isnan(k)) else k
    except Exception:
        return 0.0


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(config: dict) -> torch.device:
    if config.get("preferred_device") == "gpu" and torch.cuda.is_available():
        return torch.device(f"cuda:{config.get('nGPU', 0)}")
    return torch.device("cpu")


class SDTrainer:
    """Subject-Dependent training wrapper."""

    def __init__(self, net, config, loss_func, result_savepath=None):
        self.config = config
        self.device = resolve_device(config)
        self.net = net.to(self.device)
        self.loss_func = loss_func.to(self.device)
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.num_classes = config.get("num_classes", 4)

        self.optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["epochs"],
            eta_min=config.get("eta_min", 1e-5),
        )
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.ssr = SSRAugmentor(
            num_segs=config.get("num_segs", 8),
            num_classes=self.num_classes,
            batch_size=self.batch_size,
        )
        self.use_lr_swap = config.get("use_lr_swap", True)
        self.use_gaussian = config.get("use_gaussian_noise", True)
        self.lr_swap_prob = config.get("lr_swap_prob", 0.5)
        self.noise_sigma = config.get("noise_sigma_ratio", 0.01)
        self.noise_prob = config.get("noise_prob", 0.3)
        self.patience = config.get("early_stop_patience", 30)
        self.result_savepath = result_savepath
        if result_savepath:
            os.makedirs(result_savepath, exist_ok=True)

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 3:
            xb, yb, _ = batch
            return xb, yb
        xb, yb = batch
        return xb, yb

    def _augment_batch(self, xb, yb):
        aug_x, aug_y = self.ssr(xb, yb)
        if aug_x.numel() > 0:
            xb = torch.cat([xb.float(), aug_x.float()], dim=0)
            yb = torch.cat([yb.long(), aug_y.long()], dim=0)
        if self.use_lr_swap:
            xb = lr_hemisphere_swap(xb, p=self.lr_swap_prob)
        if self.use_gaussian:
            xb = gaussian_noise(xb, sigma_ratio=self.noise_sigma, p=self.noise_prob)
        return xb, yb

    def train_one_epoch(self, train_loader):
        self.net.train()
        all_preds, all_labels = [], []
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            xb, yb = self._unpack_batch(batch)
            xb, yb = self._augment_batch(xb, yb)
            xb = xb.float().to(self.device, non_blocking=True)
            yb = yb.long().to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.net(xb)
                loss = self.loss_func(logits, yb)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            all_preds.extend(logits.argmax(1).detach().cpu().tolist())
            all_labels.extend(yb.cpu().tolist())
            total_loss += loss.item()
            n_batches += 1
        return accuracy_score(all_labels, all_preds), total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader):
        self.net.eval()
        all_preds, all_labels = [], []
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            xb, yb = self._unpack_batch(batch)
            xb = xb.float().to(self.device, non_blocking=True)
            yb_dev = yb.long().to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.net(xb)
                loss = self.loss_func(logits, yb_dev)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(yb.tolist())
            total_loss += loss.item()
            n_batches += 1
        return {
            "acc": accuracy_score(all_labels, all_preds),
            "kappa": _safe_kappa(all_labels, all_preds),
            "f1_macro": f1_score(all_labels, all_preds, average="macro"),
            "loss": total_loss / max(n_batches, 1),
            "preds": all_preds,
            "labels": all_labels,
            "cm": confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3]),
        }

    def train(self, train_dataset, val_dataset, test_dataset):
        nw = self.config.get("num_workers", 0)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=self.use_amp,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=self.use_amp,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=self.use_amp,
        )

        best_val_acc, best_state, no_improve = 0.0, None, 0
        log_lines = []

        for epoch in range(self.epochs):
            t0 = time.time()
            tr_acc, tr_loss = self.train_one_epoch(train_loader)
            self.scheduler.step()
            val_r = self.evaluate(val_loader)
            lr_now = self.optimizer.param_groups[0]["lr"]

            if val_r["acc"] > best_val_acc:
                best_val_acc = val_r["acc"]
                best_state = copy.deepcopy(self.net.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if epoch == 0 or (epoch + 1) % 25 == 0 or no_improve == 0:
                msg = (
                    f"Ep[{epoch + 1:4d}/{self.epochs}] "
                    f"Tr={tr_acc:.4f} L={tr_loss:.4f} | "
                    f"Val={val_r['acc']:.4f} F1={val_r['f1_macro']:.4f} | "
                    f"Best={best_val_acc:.4f} LR={lr_now:.2e} "
                    f"({time.time() - t0:.1f}s)"
                )
                print(msg)
                log_lines.append(msg)

            if no_improve >= self.patience:
                print(f"  Early stop @ epoch {epoch + 1}")
                break

        if best_state is not None:
            self.net.load_state_dict(best_state)
            if self.result_savepath:
                torch.save(
                    best_state, os.path.join(self.result_savepath, "model_best.pth")
                )

        test_r = self.evaluate(test_loader)
        print(f"\n  Best Val: {best_val_acc:.6f}")
        print(
            f"  Test Acc: {test_r['acc']:.6f}  F1: {test_r['f1_macro']:.6f}  "
            f"Kappa: {test_r['kappa']:.6f}"
        )
        return test_r
