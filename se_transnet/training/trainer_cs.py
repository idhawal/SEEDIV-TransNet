from __future__ import annotations

"""
Cross-Subject (LOSO) Training Loop with Domain Adaptation.

Implements Stage 2 of the 3-stage pipeline:
  - DANN (Domain-Adversarial Neural Network) via GRL + DomainDiscriminator
  - MMD (Maximum Mean Discrepancy) multi-kernel loss
  - GRL lambda annealing: lambda = 2/(1+exp(-10*p))-1, p=epoch/total_epochs

Total loss: L = L_emotion + dann_weight * L_domain + mmd_weight * L_mmd

This is the KEY differentiator from SDTrainer — it learns domain-invariant
features that generalize across subjects by adversarially confusing the
domain classifier while maintaining emotion discrimination.
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
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

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
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(config: dict[str, Any]) -> torch.device:
    """Resolve the training device from config."""
    if config.get("preferred_device") == "gpu" and torch.cuda.is_available():
        return torch.device(f"cuda:{config.get('nGPU', 0)}")
    return torch.device("cpu")


class LOSOTrainer:
    """Cross-Subject LOSO Trainer with DANN + MMD domain adaptation.

    Unlike SDTrainer, this trainer:
      1. Uses a GradientReversalLayer + DomainDiscriminator to learn
         domain-invariant features (adversarial training)
      2. Adds MMD loss to minimize distribution distance between
         source and target features
      3. Anneals GRL lambda from 0 to 1 over training
      4. Handles larger batches (128) from pooled 14-subject data

    Args:
        net: SE-TransNet model instance.
        config: Training configuration dict.
        loss_func: Emotion classification loss (EmotionDLLoss or CE).
        num_source_subjects: Number of source subjects (14 for LOSO).
        result_savepath: Directory to save checkpoints and logs.
    """

    def __init__(
        self,
        net: nn.Module,
        config: dict[str, Any],
        loss_func: nn.Module,
        num_source_subjects: int = 14,
        result_savepath: str | None = None,
    ) -> None:
        self.config = config
        self.device = resolve_device(config)
        self.net = net.to(self.device)
        self.loss_func = loss_func.to(self.device)

        self.batch_size = config.get("batch_size", 128)
        self.epochs = config.get("epochs", 200)
        self.num_classes = config.get("num_classes", 4)

        # ── Domain adaptation components ──────────────────────────
        self.dann_weight = config.get("dann_weight", 0.1)
        self.mmd_weight = config.get("mmd_weight", 0.1)
        self.use_dann = self.dann_weight > 0
        self.use_mmd = self.mmd_weight > 0

        # GRL + Domain Discriminator
        n_domains = int(config.get("num_domains", num_source_subjects))
        self.grl = GradientReversalLayer(alpha=0.0).to(self.device)
        self.domain_disc = DomainDiscriminator(
            in_features=256,  # from SE-TransNet FC1 output
            n_domains=n_domains,
        ).to(self.device)

        # ── Optimizers ────────────────────────────────────────────
        # Joint optimizer for model + domain discriminator
        all_params = list(self.net.parameters())
        if self.use_dann:
            all_params += list(self.domain_disc.parameters())

        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("epochs", 200),
            eta_min=config.get("eta_min", 1e-5),
        )

        # ── AMP ───────────────────────────────────────────────────
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # ── Augmentation ──────────────────────────────────────────
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

        # ── Early stopping ────────────────────────────────────────
        self.patience = config.get("early_stop_patience", 40)

        # ── Logging ───────────────────────────────────────────────
        self.result_savepath = result_savepath
        if result_savepath:
            os.makedirs(result_savepath, exist_ok=True)

    def _compute_grl_lambda(self, epoch: int) -> float:
        """Anneal GRL lambda: 0 -> 1 over training.

        lambda_p = 2 / (1 + exp(-10 * p)) - 1
        where p = epoch / total_epochs

        Early epochs: lambda ~ 0 (let emotion features stabilize)
        Late epochs: lambda ~ 1 (full adversarial domain confusion)
        """
        p = epoch / max(self.epochs, 1)
        return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)

    def _augment_batch(
        self, xb: torch.Tensor, yb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations to a training batch."""
        aug_x, aug_y = self.ssr(xb, yb)
        if aug_x.numel() > 0:
            xb = torch.cat([xb.float(), aug_x.float()], dim=0)
            yb = torch.cat([yb.long(), aug_y.long()], dim=0)

        if self.use_lr_swap:
            xb = lr_hemisphere_swap(xb, p=self.lr_swap_prob)
        if self.use_gaussian:
            xb = gaussian_noise(xb, sigma_ratio=self.noise_sigma, p=self.noise_prob)
        return xb, yb

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 3:
            xb, yb, db = batch
            return xb, yb, db
        xb, yb = batch
        return xb, yb, None

    def train_one_epoch(
        self,
        train_loader: DataLoader,
        target_loader: DataLoader | None,
        epoch: int,
    ) -> dict[str, float]:
        """Train for one epoch with domain adaptation.

        Args:
            train_loader: Source (training) data loader.
            target_loader: Target (test subject) data loader for MMD.
                           If None, MMD is skipped.
            epoch: Current epoch number (for GRL lambda annealing).

        Returns:
            Dict with train_acc, train_loss, domain_loss, mmd_loss.
        """
        self.net.train()
        self.domain_disc.train()

        # Update GRL lambda
        grl_lambda = self._compute_grl_lambda(epoch)
        self.grl.set_alpha(grl_lambda)

        all_preds: list[int] = []
        all_labels: list[int] = []
        total_emotion_loss = 0.0
        total_domain_loss = 0.0
        total_mmd_loss = 0.0
        n_batches = 0

        # Optional: iterator for target data (for MMD)
        target_iter = None
        if target_loader is not None and self.use_mmd:
            target_iter = iter(target_loader)

        for batch in train_loader:
            xb, yb, db = self._unpack_batch(batch)
            xb, yb = self._augment_batch(xb, yb)
            xb = xb.float().to(self.device, non_blocking=True)
            yb = yb.long().to(self.device, non_blocking=True)
            if db is not None:
                db = db.long().to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                # Forward pass: get logits + intermediate features
                logits = self.net(xb)
                feats = self.net.extract_features(xb)  # (B, 256)

                # ── Emotion classification loss ───────────────────
                L_emotion = self.loss_func(logits, yb)

                # ── DANN domain loss ──────────────────────────────
                L_domain = torch.tensor(0.0, device=self.device)
                if self.use_dann and feats.size(0) > 1:
                    reversed_feats = self.grl(feats)
                    domain_logits = self.domain_disc(reversed_feats)
                    if db is not None:
                        L_domain = nn.functional.cross_entropy(domain_logits, db)
                    else:
                        n_domains = domain_logits.size(1)
                        domain_labels = torch.randint(
                            0,
                            n_domains,
                            (feats.size(0),),
                            device=self.device,
                        )
                        L_domain = nn.functional.cross_entropy(domain_logits, domain_labels)

                # ── MMD loss ──────────────────────────────────────
                L_mmd = torch.tensor(0.0, device=self.device)
                if self.use_mmd and target_iter is not None:
                    try:
                        tgt_xb, _ = next(target_iter)
                    except StopIteration:
                        target_iter = iter(target_loader)
                        tgt_xb, _ = next(target_iter)

                    tgt_xb = tgt_xb.float().to(self.device, non_blocking=True)
                    with torch.no_grad():
                        tgt_feats = self.net.extract_features(tgt_xb)
                    L_mmd = compute_mmd(feats, tgt_feats.detach())

                # ── Total loss ────────────────────────────────────
                loss = L_emotion + self.dann_weight * L_domain + self.mmd_weight * L_mmd

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            if self.use_dann:
                torch.nn.utils.clip_grad_norm_(
                    self.domain_disc.parameters(), max_norm=1.0
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            all_preds.extend(logits.argmax(1).detach().cpu().tolist())
            all_labels.extend(yb.cpu().tolist())
            total_emotion_loss += L_emotion.item()
            total_domain_loss += L_domain.item()
            total_mmd_loss += L_mmd.item()
            n_batches += 1

        acc = accuracy_score(all_labels, all_preds)
        return {
            "train_acc": acc,
            "train_loss": total_emotion_loss / max(n_batches, 1),
            "domain_loss": total_domain_loss / max(n_batches, 1),
            "mmd_loss": total_mmd_loss / max(n_batches, 1),
            "grl_lambda": grl_lambda,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        """Evaluate on a dataset.

        Returns:
            Dict with acc, kappa, f1_macro, loss, preds, labels, cm.
        """
        self.net.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            xb, yb, _ = self._unpack_batch(batch)
            xb = xb.float().to(self.device, non_blocking=True)
            yb_dev = yb.long().to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.net(xb)
                loss = self.loss_func(logits, yb_dev)

            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(yb.tolist())
            total_loss += loss.item()
            n_batches += 1

        acc = accuracy_score(all_labels, all_preds)
        kappa = _safe_kappa(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])

        return {
            "acc": acc,
            "kappa": kappa,
            "f1_macro": f1,
            "loss": total_loss / max(n_batches, 1),
            "preds": all_preds,
            "labels": all_labels,
            "cm": cm,
        }

    def train(
        self,
        train_dataset: SeedIVDataset,
        val_dataset: SeedIVDataset,
        test_dataset: SeedIVDataset,
    ) -> dict[str, Any]:
        """Full LOSO training loop with DANN+MMD and early stopping.

        Args:
            train_dataset: Source subjects training data.
            val_dataset: Validation split (from source subjects).
            test_dataset: Target subject data (held out).

        Returns:
            Dict with final test metrics.
        """
        num_workers = self.config.get("num_workers", 0)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.use_amp,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.use_amp,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # shuffle for MMD sampling
            num_workers=num_workers,
            pin_memory=self.use_amp,
            drop_last=False,
        )

        # Target loader for MMD (uses test data WITHOUT labels)
        target_loader = test_loader if self.use_mmd else None

        best_val_acc = 0.0
        best_model_state = None
        epochs_no_improve = 0
        log_lines: list[str] = []

        for epoch in range(self.epochs):
            t0 = time.time()

            # Train with domain adaptation
            train_metrics = self.train_one_epoch(train_loader, target_loader, epoch)
            self.scheduler.step()

            # Validate
            val_result = self.evaluate(val_loader)
            val_acc = val_result["acc"]
            lr_now = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            # Best model tracking
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(self.net.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Logging (every 25 epochs or on improvement)
            if epoch == 0 or (epoch + 1) % 25 == 0 or epochs_no_improve == 0:
                msg = (
                    f"Ep[{epoch + 1:4d}/{self.epochs}] "
                    f"Tr={train_metrics['train_acc']:.4f} "
                    f"L_e={train_metrics['train_loss']:.4f} "
                    f"L_d={train_metrics['domain_loss']:.4f} "
                    f"L_m={train_metrics['mmd_loss']:.4f} | "
                    f"Val={val_acc:.4f} "
                    f"F1={val_result['f1_macro']:.4f} | "
                    f"Best={best_val_acc:.4f} "
                    f"GRL={train_metrics['grl_lambda']:.3f} "
                    f"LR={lr_now:.2e} ({elapsed:.1f}s)"
                )
                print(msg)
                log_lines.append(msg)

            # Early stopping
            if epochs_no_improve >= self.patience:
                print(
                    f"  Early stop @ epoch {epoch + 1} "
                    f"(no improvement for {self.patience} epochs)"
                )
                break

        # ── Final evaluation with best model ──────────────────────
        if best_model_state is not None:
            self.net.load_state_dict(best_model_state)
            if self.result_savepath:
                ckpt_path = os.path.join(self.result_savepath, "model_best.pth")
                torch.save(best_model_state, ckpt_path)

        test_result = self.evaluate(test_loader)

        print(f"\n  Best Val Accuracy : {best_val_acc:.6f}")
        print(f"  Test Accuracy    : {test_result['acc']:.6f}")
        print(f"  Test F1-macro    : {test_result['f1_macro']:.6f}")
        print(f"  Test Kappa       : {test_result['kappa']:.6f}")

        # Save log
        if self.result_savepath:
            log_path = os.path.join(self.result_savepath, "training_log.txt")
            with open(log_path, "w") as f:
                f.write("\n".join(log_lines))
                f.write(
                    f"\n\nBest Val Acc: {best_val_acc:.6f}\n"
                    f"Test Acc: {test_result['acc']:.6f}\n"
                    f"Test F1: {test_result['f1_macro']:.6f}\n"
                    f"Test Kappa: {test_result['kappa']:.6f}\n"
                )

        return test_result
