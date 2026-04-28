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
from se_transnet.data.augmentation import (
    SSRAugmentor,
    gaussian_noise,
    lr_hemisphere_swap,
)
from se_transnet.data.datasets import SeedIVDataset
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader


def _safe_kappa(y_true, y_pred):
    import math

    try:
        k = cohen_kappa_score(y_true, y_pred)
        return 0.0 if (isinstance(k, float) and math.isnan(k)) else k
    except Exception:
        return 0.0


def set_seed(seed: int = 42, benchmark: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Unlocking benchmark=True allows CUDNN to auto-tune convolutional operations
        torch.backends.cudnn.deterministic = not benchmark
        torch.backends.cudnn.benchmark = benchmark


class SDTrainer:
    def __init__(
        self,
        net: nn.Module,
        config: dict,
        loss_func: nn.Module,
        result_savepath: str | None = None,
    ) -> None:
        self.net = net
        self.config = config
        self.loss_func = loss_func
        self.result_savepath = result_savepath

        dev_str = config.get("preferred_device", "cpu")
        if dev_str == "gpu" and torch.cuda.is_available():
            dev_id = config.get("nGPU", 0)
            self.device = torch.device(f"cuda:{dev_id}")
        else:
            self.device = torch.device("cpu")

        self.net.to(self.device)
        self.loss_func.to(self.device)

        self.epochs = config.get("epochs", 200)
        self.batch_size = config.get("batch_size", 32)
        self.patience = config.get("early_stop_patience", 30)

        self.optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=config.get("eta_min", 1e-5)
        )
        self.scaler = (
            torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None
        )

        if config.get("num_segs", 8) > 0:
            self.ssr_aug = SSRAugmentor(
                num_segs=config.get("num_segs", 8),
                num_classes=config.get("num_classes", 4),
                batch_size=self.batch_size,
            )
        else:
            self.ssr_aug = None

        if self.result_savepath:
            os.makedirs(self.result_savepath, exist_ok=True)

    def train_one_epoch(self, train_loader: DataLoader) -> tuple:
        self.net.train()
        total_loss, correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            if self.ssr_aug is not None:
                aug_x, aug_y = self.ssr_aug(xb, yb)
                if len(aug_y) > 0:
                    xb = torch.cat([xb, aug_x], dim=0)
                    yb = torch.cat([yb, aug_y], dim=0)

            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)

            if self.config.get("use_lr_swap", False):
                xb = lr_hemisphere_swap(xb, p=self.config.get("lr_swap_prob", 0.5))
            if self.config.get("use_gaussian_noise", False):
                xb = gaussian_noise(
                    xb,
                    sigma_ratio=self.config.get("noise_sigma_ratio", 0.01),
                    p=self.config.get("noise_prob", 0.3),
                )

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = self.net(xb)
                    loss = self.loss_func(logits, yb)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.net(xb)
                loss = self.loss_func(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.net.eval()
        all_preds, all_labels = [], []

        for batch in loader:
            xb = batch[0].to(self.device, non_blocking=True)
            yb = batch[1]
            logits = self.net(xb)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(yb.numpy())

        return {
            "acc": accuracy_score(all_labels, all_preds),
            "f1_macro": f1_score(
                all_labels, all_preds, average="macro", zero_division=0
            ),
            "kappa": _safe_kappa(all_labels, all_preds),
            "cm": confusion_matrix(all_labels, all_preds, labels=list(range(4))),
        }

    def train(
        self, train_ds: SeedIVDataset, val_ds: SeedIVDataset, test_ds: SeedIVDataset
    ) -> dict:
        nw = self.config.get("num_workers", 0)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=nw,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False, num_workers=nw
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False, num_workers=nw
        )

        best_val_acc = 0.0
        best_state = None
        no_improve = 0
        log_lines = []

        t0 = time.time()
        for epoch in range(self.epochs):
            tr_loss, tr_acc = self.train_one_epoch(train_loader)
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

        test_result = self.evaluate(test_loader)
        return test_result
