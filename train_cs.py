from __future__ import annotations

"""
Cross-Subject (LOSO) Training Loop with True Domain Adaptation.

Implementing Unsupervised Domain Adaptation (UDA):
  - DANN (Binary Domain-Adversarial Neural Network):
    Aligns the unlabeled Test subject (Target) directly with the pooled Train subjects (Source)
  - MMD (Maximum Mean Discrepancy) multi-kernel loss between Source and Target
  - GRL lambda annealing: lambda = 2/(1+exp(-10*p))-1, p=epoch/total_epochs
"""

import copy
import math
import os
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from se_transnet.data.augmentation import (
    SSRAugmentor,
    gaussian_noise,
    lr_hemisphere_swap,
)
from se_transnet.data.datasets import SeedIVDataset
from se_transnet.training.losses import (
    DomainDiscriminator,
    GradientReversalLayer,
    compute_mmd,
)
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


class LOSOTrainer:
    def __init__(
        self,
        net: nn.Module,
        config: dict,
        loss_func: nn.Module,
        num_source_subjects: int = 14,  # Left for interface compatibility
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

        # Binary Discriminator for Source (0) vs Target (1)
        self.domain_disc = DomainDiscriminator(in_features=256, n_domains=2).to(
            self.device
        )
        self.grl = GradientReversalLayer(alpha=0.0)

        self.dann_weight = config.get("dann_weight", 0.1)
        self.mmd_weight = config.get("mmd_weight", 0.1)

        self.epochs = config.get("epochs", 200)
        self.batch_size = config.get("batch_size", 128)
        self.patience = config.get("early_stop_patience", 40)

        # Optimize Network AND Discriminator together
        params = list(self.net.parameters()) + list(self.domain_disc.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
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

    def _compute_grl_lambda(self, epoch: int) -> float:
        """Ganin et al. (2015) scheduling schedule 0 -> 1."""
        if self.epochs == 0:
            return 1.0
        p = epoch / self.epochs
        return (2.0 / (1.0 + math.exp(-10.0 * p))) - 1.0

    def train_one_epoch(
        self, train_loader: DataLoader, target_loader: DataLoader, epoch: int
    ) -> tuple:
        self.net.train()
        self.domain_disc.train()

        # Update GRL
        current_lambda = self._compute_grl_lambda(epoch)
        self.grl.set_alpha(current_lambda)

        total_cls_loss, total_dann_loss, total_mmd_loss = 0.0, 0.0, 0.0
        correct_cls, total_samples = 0, 0

        target_iter = iter(target_loader)

        for src_batch in train_loader:
            src_xb = src_batch[0]
            src_yb = src_batch[1]

            if self.ssr_aug is not None:
                aug_x, aug_y = self.ssr_aug(src_xb, src_yb)
                if len(aug_y) > 0:
                    src_xb = torch.cat([src_xb, aug_x], dim=0)
                    src_yb = torch.cat([src_yb, aug_y], dim=0)

            src_xb = src_xb.to(self.device, non_blocking=True)
            src_yb = src_yb.to(self.device, non_blocking=True)

            # Fetch Unlabeled Target Batch
            try:
                tgt_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tgt_batch = next(target_iter)
            tgt_xb = tgt_batch[0].to(self.device, non_blocking=True)

            if self.config.get("use_lr_swap", False):
                p_swap = self.config.get("lr_swap_prob", 0.5)
                src_xb = lr_hemisphere_swap(src_xb, p=p_swap)
                tgt_xb = lr_hemisphere_swap(tgt_xb, p=p_swap)
            if self.config.get("use_gaussian_noise", False):
                p_noise = self.config.get("noise_prob", 0.3)
                sig_ratio = self.config.get("noise_sigma_ratio", 0.01)
                src_xb = gaussian_noise(src_xb, sigma_ratio=sig_ratio, p=p_noise)
                tgt_xb = gaussian_noise(tgt_xb, sigma_ratio=sig_ratio, p=p_noise)

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    # Source forward
                    src_feats = self.net.extract_features(src_xb)
                    src_logits = self.net.classifier(src_feats)
                    loss_cls = self.loss_func(src_logits, src_yb)

                    # DANN
                    if self.dann_weight > 0:
                        tgt_feats = self.net.extract_features(tgt_xb)
                        comb_feats = torch.cat([src_feats, tgt_feats], dim=0)
                        domain_preds = self.domain_disc(self.grl(comb_feats))

                        domain_labels = torch.cat(
                            [
                                torch.zeros(
                                    src_xb.size(0), dtype=torch.long, device=self.device
                                ),
                                torch.ones(
                                    tgt_xb.size(0), dtype=torch.long, device=self.device
                                ),
                            ]
                        )
                        loss_dann = F.cross_entropy(domain_preds, domain_labels)
                    else:
                        loss_dann = torch.tensor(0.0, device=self.device)
                        tgt_feats = src_feats  # Fallback

                    # MMD
                    if self.mmd_weight > 0:
                        if self.dann_weight == 0:
                            tgt_feats = self.net.extract_features(tgt_xb)
                        loss_mmd = compute_mmd(src_feats, tgt_feats)
                    else:
                        loss_mmd = torch.tensor(0.0, device=self.device)

                    total_loss = (
                        loss_cls
                        + (self.dann_weight * loss_dann)
                        + (self.mmd_weight * loss_mmd)
                    )

                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # Same operations without AMP
                src_feats = self.net.extract_features(src_xb)
                src_logits = self.net.classifier(src_feats)
                loss_cls = self.loss_func(src_logits, src_yb)

                if self.dann_weight > 0:
                    tgt_feats = self.net.extract_features(tgt_xb)
                    domain_preds = self.domain_disc(
                        self.grl(torch.cat([src_feats, tgt_feats], dim=0))
                    )
                    domain_labels = torch.cat(
                        [
                            torch.zeros(
                                src_xb.size(0), dtype=torch.long, device=self.device
                            ),
                            torch.ones(
                                tgt_xb.size(0), dtype=torch.long, device=self.device
                            ),
                        ]
                    )
                    loss_dann = F.cross_entropy(domain_preds, domain_labels)
                else:
                    loss_dann = torch.tensor(0.0, device=self.device)
                    tgt_feats = src_feats

                if self.mmd_weight > 0:
                    if self.dann_weight == 0:
                        tgt_feats = self.net.extract_features(tgt_xb)
                    loss_mmd = compute_mmd(src_feats, tgt_feats)
                else:
                    loss_mmd = torch.tensor(0.0, device=self.device)

                total_loss = (
                    loss_cls
                    + (self.dann_weight * loss_dann)
                    + (self.mmd_weight * loss_mmd)
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Tracking
            bs = src_xb.size(0)
            total_cls_loss += loss_cls.item() * bs
            total_dann_loss += loss_dann.item() * bs
            total_mmd_loss += loss_mmd.item() * bs
            preds = src_logits.argmax(dim=1)
            correct_cls += (preds == src_yb).sum().item()
            total_samples += bs

        return (
            total_cls_loss / total_samples,
            total_dann_loss / total_samples,
            total_mmd_loss / total_samples,
            correct_cls / total_samples,
            current_lambda,
        )

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
        # Using the test subject's data as the unlabeled Target Domain
        unlabeled_target_loader = DataLoader(
            test_ds,
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
        best_model_state = None
        epochs_no_improve = 0
        log_lines = []
        t0 = time.time()

        for epoch in range(self.epochs):
            (l_cls, l_dann, l_mmd, acc_tr, cur_lam) = self.train_one_epoch(
                train_loader, unlabeled_target_loader, epoch
            )
            self.scheduler.step()

            val_res = self.evaluate(val_loader)
            val_acc = val_res["acc"]

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(self.net.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch == 0 or (epoch + 1) % 10 == 0 or epochs_no_improve == 0:
                msg = (
                    f"Ep[{epoch + 1:3d}/{self.epochs}] "
                    f"Tr_Acc={acc_tr:.3f} Lc={l_cls:.3f} Ld={l_dann:.3f} Lm={l_mmd:.3f} "
                    f"| Val_Acc={val_acc:.3f} (Best={best_val_acc:.3f}) | "
                    f"lam={cur_lam:.2f} ({time.time() - t0:.1f}s)"
                )
                print(msg)
                log_lines.append(msg)

            if epochs_no_improve >= self.patience:
                print(f"  Early stop @ epoch {epoch + 1} ")
                break

        if best_model_state is not None:
            self.net.load_state_dict(best_model_state)
            if self.result_savepath:
                torch.save(
                    best_model_state,
                    os.path.join(self.result_savepath, "model_best.pth"),
                )

        test_result = self.evaluate(test_loader)
        return test_result
