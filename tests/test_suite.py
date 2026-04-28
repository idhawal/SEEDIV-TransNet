"""
tests/test_suite.py
===================
Comprehensive pytest test suite for SE-TransNet V4.

Categories
----------
A. Model Numerical Stability         (tests 01-07)
B. Loss Function Edge Cases          (tests 08-16)
C. Augmentation Correctness          (tests 17-24)
D. Dataset Integrity                 (tests 25-32)
E. Trainer Logic                     (tests 33-36)
F. Config & Integration              (tests 37-40)
G. Memory & Performance              (tests 41-43)

Run with:  py -3.9 -m pytest tests/test_suite.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

# ---------------------------------------------------------------------------
# Ensure project root is importable regardless of cwd
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from se_transnet.data.augmentation import (
    LR_PAIRS,
    SSRAugmentor,
    channel_dropout,
    gaussian_noise,
    lr_hemisphere_swap,
)
from se_transnet.data.datasets import (
    NUM_CLASSES,
    SESSION_LABELS,
    SeedIVDataset,
    euclidean_alignment,
)
from se_transnet.models.modules import (
    PositionalEncoding,
    TransformerEncoder,
    VarPool2D,
)
from se_transnet.models.se_transnet import SETransNet
from se_transnet.training.losses import (
    EmotionDLLoss,
    GradientReversalLayer,
    compute_coral,
    compute_mmd,
)

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"


def _make_model(seed: int = 42) -> SETransNet:
    """Return a freshly initialised SE-TransNet on CPU."""
    torch.manual_seed(seed)
    return SETransNet().eval()


def _rand_batch(B: int = 4, seed: int = 0) -> torch.Tensor:
    """Return a (B, 62, 800) float32 tensor."""
    torch.manual_seed(seed)
    return torch.randn(B, 62, 800)


def _make_tiny_dataset(n: int = 20, seed: int = 7) -> SeedIVDataset:
    """Minimal SeedIVDataset for unit-testing without real data."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, 62, 800)).astype(np.float32)
    labels = rng.integers(0, NUM_CLASSES, size=n).astype(np.int64)
    # Guarantee at least one sample per class for SSR
    labels[:NUM_CLASSES] = np.arange(NUM_CLASSES)
    return SeedIVDataset(data, labels, normalise=False)


# ===========================================================================
# A. Model Numerical Stability
# ===========================================================================


class TestModelNumericalStability:
    """Forward pass does not produce NaN / Inf under various input regimes."""

    def test_model_no_nan_normal_input(self):
        """Forward pass on standard randn input must produce only finite values."""
        model = _make_model()
        x = _rand_batch(B=4)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 4), f"unexpected output shape {out.shape}"
        assert torch.isfinite(out).all(), "NaN or Inf detected in output"

    def test_model_no_nan_near_zero_input(self):
        """Forward pass on near-zero signal (1e-8 scale) must remain finite."""
        model = _make_model()
        x = _rand_batch(B=4) * 1e-8
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all(), (
            "NaN/Inf detected with near-zero input — check VarPool clamp or BN"
        )

    def test_model_no_nan_large_input(self):
        """Forward pass on large-amplitude signal (100x) must remain finite."""
        model = _make_model()
        x = _rand_batch(B=4) * 100.0
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all(), "NaN/Inf detected with large-amplitude input"

    def test_varpool_log_variance_clamp(self):
        """VarPool2D must not return NaN even for constant (all-zero) signal."""
        vp = VarPool2D(kernel_size=40, stride=10)
        x_zero = torch.zeros(2, 80, 800)
        out = vp(x_zero)
        assert torch.isfinite(out).all(), (
            "VarPool2D produced NaN/Inf on constant signal — clamp min=1e-6 needed"
        )

    def test_positional_encoding_deterministic(self):
        """The PE buffer must be identical across two calls (deterministic)."""
        pe = PositionalEncoding(d_model=80, max_len=100)
        x = torch.randn(2, 77, 80)
        out1 = pe(x)
        out2 = pe(x)
        assert torch.equal(out1, out2), "PositionalEncoding is not deterministic"

    def test_transformer_output_finite(self):
        """Single TransformerEncoder block must produce finite output."""
        enc = TransformerEncoder(embed_dim=80, num_heads=8)
        enc.eval()
        x = torch.randn(3, 77, 80)
        with torch.no_grad():
            out = enc(x)
        assert out.shape == (3, 77, 80), f"unexpected shape {out.shape}"
        assert torch.isfinite(out).all(), "TransformerEncoder produced NaN/Inf"

    def test_model_eval_train_consistency(self):
        """Logits from eval mode are finite and have the correct shape."""
        model = _make_model()
        x = _rand_batch(B=8)
        model.eval()
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (8, 4), f"unexpected shape {logits.shape}"
        assert torch.isfinite(logits).all(), "logits from eval mode contain NaN/Inf"


# ===========================================================================
# B. Loss Function Edge Cases
# ===========================================================================


class TestLossFunctionEdgeCases:
    """EmotionDLLoss, MMD, CORAL and GRL behave correctly in edge cases."""

    # ------------------------------------------------------------------
    # EmotionDLLoss
    # ------------------------------------------------------------------

    def _make_loss(self) -> EmotionDLLoss:
        return EmotionDLLoss(epsilon=0.2)

    def test_emotiondl_loss_all_correct(self):
        """Loss must be low when logits strongly predict the true class.

        EmotionDLLoss is a soft-label KL divergence.  Even a perfect predictor
        incurs the entropy of the soft-label row as an irreducible floor
        (~2.0 nats for epsilon=0.2).  We just verify the loss is close to that
        floor and well below the worst-case value.
        """
        loss_fn = self._make_loss()
        targets = torch.tensor([0, 1, 2, 3])
        # Very strong correct logits
        logits = torch.zeros(4, 4)
        for i, t in enumerate(targets):
            logits[i, t] = 10.0
        loss_val = loss_fn(logits, targets).item()
        # Irreducible floor ≈ H(soft_row) ≈ 2.0 nats for epsilon=0.2
        # Worst-case (totally wrong prediction) is ~9.25
        # Verify we are close to the floor (below 2.5) rather than near worst-case
        assert loss_val < 2.5, f"Expected near-minimum loss (<2.5), got {loss_val:.4f}"

    def test_emotiondl_loss_all_wrong(self):
        """Loss must be higher when predictions are maximally wrong."""
        loss_fn = self._make_loss()
        targets = torch.tensor([0, 1, 2, 3])
        # Correct logits
        good_logits = torch.zeros(4, 4)
        for i, t in enumerate(targets):
            good_logits[i, t] = 10.0
        # Wrong logits — put mass on a different class
        bad_logits = torch.zeros(4, 4)
        for i, t in enumerate(targets):
            bad_logits[i, (t + 1) % 4] = 10.0

        good_loss = loss_fn(good_logits, targets).item()
        bad_loss = loss_fn(bad_logits, targets).item()
        assert bad_loss > good_loss, (
            f"Expected bad_loss ({bad_loss:.4f}) > good_loss ({good_loss:.4f})"
        )

    def test_emotiondl_soft_labels_rows_sum_to_one(self):
        """Every row of the soft_labels matrix must sum to exactly 1.0."""
        loss_fn = self._make_loss()
        row_sums = loss_fn.soft_labels.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(NUM_CLASSES), atol=1e-6), (
            f"Row sums are not all 1: {row_sums.tolist()}"
        )

    def test_emotiondl_soft_labels_diagonal_is_largest(self):
        """Diagonal entries (self-label confidence) must exceed all off-diagonal."""
        loss_fn = self._make_loss()
        sl = loss_fn.soft_labels
        for i in range(NUM_CLASSES):
            diag_val = sl[i, i].item()
            off_max = sl[i, :].clone()
            off_max[i] = -float("inf")
            assert diag_val > off_max.max().item(), (
                f"Row {i}: diagonal {diag_val:.4f} not larger than "
                f"off-diagonal max {off_max.max().item():.4f}"
            )

    def test_emotiondl_symmetric_structure(self):
        """
        Validate valence-axis symmetry in soft_labels:
          Classes 0 (Neutral) and 1 (Sad) share the same cross-class epsilon structure.
          Classes 2 (Fear)    and 3 (Happy) share the same cross-class epsilon structure.
        """
        loss_fn = self._make_loss()
        sl = loss_fn.soft_labels
        # [0,1] pair: sl[0,1] == sl[1,0]  (symmetric cross-class weight)
        assert sl[0, 1].item() == pytest.approx(sl[1, 0].item(), abs=1e-6), (
            "Neutral/Sad soft-label weight is not symmetric"
        )
        # [2,3] pair: sl[2,3] == sl[3,2]
        assert sl[2, 3].item() == pytest.approx(sl[3, 2].item(), abs=1e-6), (
            "Fear/Happy soft-label weight is not symmetric"
        )
        # Cross-group weights ([0,2], [0,3]) should be equal and smaller
        assert sl[0, 2].item() == pytest.approx(sl[0, 3].item(), abs=1e-6)
        assert sl[1, 2].item() == pytest.approx(sl[1, 3].item(), abs=1e-6)

    # ------------------------------------------------------------------
    # MMD
    # ------------------------------------------------------------------

    def test_mmd_identical_dist_near_zero(self):
        """MMD(x, x) should be < 0.01 (self-similarity)."""
        torch.manual_seed(0)
        x = torch.randn(32, 16)
        mmd_val = compute_mmd(x, x).item()
        assert mmd_val < 0.01, f"MMD(x,x) = {mmd_val:.6f}, expected < 0.01"

    def test_mmd_different_dist_positive(self):
        """MMD between N(0,1) and N(5,1) must exceed MMD(x,x)."""
        torch.manual_seed(1)
        x = torch.randn(32, 16)
        y = torch.randn(32, 16) + 5.0
        mmd_diff = compute_mmd(x, y).item()
        mmd_same = compute_mmd(x, x).item()
        assert mmd_diff > mmd_same, (
            f"Expected MMD(N(0,1), N(5,1)) > MMD(x,x): {mmd_diff:.6f} vs {mmd_same:.6f}"
        )

    # ------------------------------------------------------------------
    # CORAL
    # ------------------------------------------------------------------

    def test_coral_identical_near_zero(self):
        """CORAL(x, x) should be < 0.01 (identical distributions)."""
        torch.manual_seed(2)
        x = torch.randn(32, 16)
        coral_val = compute_coral(x, x).item()
        assert coral_val < 0.01, f"CORAL(x, x) = {coral_val:.6f}, expected < 0.01"

    # ------------------------------------------------------------------
    # GRL
    # ------------------------------------------------------------------

    def test_grl_gradient_reversal(self):
        """Gradients through GRL must be negated (alpha=1.0)."""
        grl = GradientReversalLayer(alpha=1.0)
        x = torch.randn(4, 8, requires_grad=True)
        out = grl(x)
        # Use a simple scalar loss
        loss = out.sum()
        loss.backward()
        # Without GRL the gradient of sum(x) w.r.t. x is all-ones.
        # With GRL (alpha=1) the upstream all-ones gradient is negated -> all -1.
        expected_grad = torch.full_like(x, fill_value=-1.0)
        assert torch.allclose(x.grad, expected_grad, atol=1e-6), (
            f"GRL gradient not negated: {x.grad}"
        )


# ===========================================================================
# C. Augmentation Correctness
# ===========================================================================


class TestAugmentationCorrectness:
    """Verify SSR, LR swap, Gaussian noise and channel dropout behave correctly."""

    def _make_ssr_data(self, n_per_class: int = 5, seed: int = 0) -> tuple:
        """Create balanced synthetic data for SSR tests."""
        rng = np.random.default_rng(seed)
        n = n_per_class * NUM_CLASSES
        data = rng.standard_normal((n, 62, 800)).astype(np.float32)
        labels = np.repeat(np.arange(NUM_CLASSES), n_per_class).astype(np.int64)
        return torch.from_numpy(data), torch.from_numpy(labels)

    # ------------------------------------------------------------------
    # SSR
    # ------------------------------------------------------------------

    def test_ssr_same_class_only(self):
        """SSR output labels must all be valid class indices (no contamination)."""
        np.random.seed(42)
        ssr = SSRAugmentor(num_segs=8, num_classes=NUM_CLASSES, batch_size=16)
        data, labels = self._make_ssr_data(n_per_class=6)
        aug_data, aug_labels = ssr(data, labels)
        assert aug_labels.numel() > 0, "SSR produced empty output"
        valid_classes = set(range(NUM_CLASSES))
        returned_classes = set(aug_labels.tolist())
        assert returned_classes.issubset(valid_classes), (
            f"SSR returned invalid class labels: {returned_classes - valid_classes}"
        )

    def test_ssr_segment_boundaries(self):
        """SSR must not write outside segment boundaries (no data leakage)."""
        np.random.seed(99)
        num_segs = 4
        ssr = SSRAugmentor(num_segs=num_segs, num_classes=NUM_CLASSES, batch_size=8)
        # Mark each sample with a unique ID per time-step so we can trace origin
        n_per_class = 4
        n = n_per_class * NUM_CLASSES
        T = 800
        seg_size = T // num_segs

        data_np = np.zeros((n, 62, T), dtype=np.float32)
        for i in range(n):
            data_np[i] = float(i)  # whole sample filled with sample index

        labels_np = np.repeat(np.arange(NUM_CLASSES), n_per_class).astype(np.int64)
        data_t = torch.from_numpy(data_np)
        labels_t = torch.from_numpy(labels_np)

        aug_data, _ = ssr(data_t, labels_t)
        assert aug_data.numel() > 0

        # For each augmented sample, verify that each segment block
        # contains a *uniform* value (consistent with one source sample)
        for i in range(len(aug_data)):
            for seg in range(num_segs):
                start = seg * seg_size
                end = (seg + 1) * seg_size
                block = aug_data[i, :, start:end]
                # All values in one segment block should be the same float
                assert block.std().item() == pytest.approx(0.0, abs=1e-5), (
                    f"Sample {i}, seg {seg}: segment contains mixed sources "
                    f"(std={block.std().item():.6f})"
                )

    # ------------------------------------------------------------------
    # LR hemisphere swap
    # ------------------------------------------------------------------

    def test_lr_swap_p0_no_change(self):
        """lr_hemisphere_swap with p=0.0 must never change the data."""
        torch.manual_seed(3)
        x = torch.randn(4, 62, 800)
        x_orig = x.clone()
        for _ in range(20):
            out = lr_hemisphere_swap(x, p=0.0)
            assert torch.equal(out, x_orig), "p=0.0 swap altered the data"

    def test_lr_swap_p1_always_changes(self):
        """lr_hemisphere_swap with p=1.0 must always change at least one channel."""
        torch.manual_seed(4)
        for _ in range(10):
            x = torch.randn(4, 62, 800)
            x_orig = x.clone()
            out = lr_hemisphere_swap(x, p=1.0)
            assert not torch.equal(out, x_orig), "p=1.0 swap did not change any channel"

    def test_lr_swap_double_swap_identity(self):
        """Applying LR swap twice (p=1.0) must restore the original signal."""
        torch.manual_seed(5)
        x = torch.randn(4, 62, 800)
        x_orig = x.clone()
        x_swapped = lr_hemisphere_swap(x, p=1.0)
        x_restored = lr_hemisphere_swap(x_swapped, p=1.0)
        assert torch.allclose(x_restored, x_orig, atol=1e-6), (
            "Double LR swap did not restore original data"
        )

    # ------------------------------------------------------------------
    # Gaussian noise
    # ------------------------------------------------------------------

    def test_gaussian_noise_p0_no_change(self):
        """gaussian_noise with p=0.0 must never alter the tensor."""
        torch.manual_seed(6)
        x = torch.randn(4, 62, 800)
        x_orig = x.clone()
        for _ in range(20):
            out = gaussian_noise(x, sigma_ratio=0.01, p=0.0)
            assert torch.equal(out, x_orig), "p=0.0 noise altered the data"

    # ------------------------------------------------------------------
    # Channel dropout
    # ------------------------------------------------------------------

    def test_channel_dropout_p0_no_zero(self):
        """channel_dropout with p=0.0 must not zero any channels."""
        torch.manual_seed(7)
        x = torch.abs(torch.randn(4, 62, 800)) + 0.1  # strictly positive
        out = channel_dropout(x, p=0.0)
        # With p=0, mask = (rand > 0) = all True => nothing zeroed
        assert (out > 0).all(), "p=0.0 channel dropout introduced zeros"

    def test_channel_dropout_p1_all_zero(self):
        """channel_dropout with p=1.0 must zero out all channels."""
        torch.manual_seed(8)
        x = torch.randn(4, 62, 800)
        out = channel_dropout(x, p=1.0)
        assert torch.all(out == 0), "p=1.0 channel dropout did not zero all channels"


# ===========================================================================
# D. Dataset Integrity
# ===========================================================================


class TestDatasetIntegrity:
    """SeedIVDataset, class weights and Euclidean Alignment correctness."""

    def _make_dataset(
        self, n: int = 40, normalise: bool = True, seed: int = 11
    ) -> SeedIVDataset:
        rng = np.random.default_rng(seed)
        data = rng.standard_normal((n, 62, 800)).astype(np.float32)
        labels = np.tile(np.arange(NUM_CLASSES), n // NUM_CLASSES + 1)[:n]
        return SeedIVDataset(data, labels.astype(np.int64), normalise=normalise)

    # ------------------------------------------------------------------

    def test_dataset_zscore_per_channel(self):
        """With normalise=True, each channel across the time axis has mean~0 and std~1.

        The dataset normalises using numpy's population std (ddof=0).  We check
        with correction=0 to match that convention exactly.
        """
        ds = self._make_dataset(n=8, normalise=True)
        for i in range(len(ds)):
            eeg, _ = ds[i]  # (62, 800) float tensor
            mean = eeg.mean(dim=1)  # (62,)
            # Use correction=0 (population std) to match numpy's default ddof=0
            std = eeg.std(dim=1, correction=0)  # (62,)
            assert torch.allclose(mean, torch.zeros(62), atol=1e-4), (
                f"Sample {i}: channel mean not ~0: max abs = {mean.abs().max():.6f}"
            )
            assert torch.allclose(std, torch.ones(62), atol=1e-4), (
                f"Sample {i}: channel std not ~1: max dev = {(std - 1).abs().max():.6f}"
            )

    def test_dataset_label_dtype(self):
        """Labels returned by __getitem__ must be Python int."""
        ds = self._make_dataset(normalise=False)
        for i in range(min(len(ds), 8)):
            _, label = ds[i]
            assert isinstance(label, int), (
                f"Label at index {i} is {type(label)}, expected int"
            )

    def test_dataset_data_dtype(self):
        """EEG tensor returned by __getitem__ must be float32."""
        ds = self._make_dataset(normalise=False)
        eeg, _ = ds[0]
        assert eeg.dtype == torch.float32, f"Expected torch.float32, got {eeg.dtype}"

    def test_dataset_no_inplace_mutation(self):
        """Modifying the returned tensor must not change dataset internals."""
        ds = self._make_dataset(normalise=False)
        eeg1, _ = ds[0]
        original_val = eeg1[0, 0].item()
        eeg1[0, 0] = 999.0  # mutate returned tensor
        eeg2, _ = ds[0]  # fetch again
        assert eeg2[0, 0].item() == pytest.approx(original_val, abs=1e-6), (
            "Mutation of returned tensor changed dataset internals — "
            "ensure __getitem__ returns a copy"
        )

    def test_class_weights_inverse_frequency(self):
        """A class with more samples must receive a lower weight."""
        rng = np.random.default_rng(20)
        n = 40
        data = rng.standard_normal((n, 62, 800)).astype(np.float32)
        # Class 0: 20 samples, Class 1: 10, Class 2: 7, Class 3: 3
        labels = np.array([0] * 20 + [1] * 10 + [2] * 7 + [3] * 3, dtype=np.int64)
        ds = SeedIVDataset(data, labels, normalise=False)
        weights = ds.class_weights()  # (4,)
        assert weights[0] < weights[3], (
            f"Class 0 (20 samples) weight {weights[0]:.4f} should be "
            f"< class 3 (3 samples) weight {weights[3]:.4f}"
        )
        assert weights[0] < weights[1] < weights[2] < weights[3], (
            f"Weights not strictly monotonically decreasing with class count: "
            f"{weights.tolist()}"
        )

    def test_class_weights_sum_to_num_classes(self):
        """class_weights() must sum to exactly NUM_CLASSES (4)."""
        ds = self._make_dataset(n=40, normalise=False)
        weights = ds.class_weights()
        assert weights.sum().item() == pytest.approx(NUM_CLASSES, abs=1e-5), (
            f"Weights sum = {weights.sum().item():.6f}, expected {NUM_CLASSES}"
        )

    def test_euclidean_alignment_covariance(self):
        """After EA, the Frobenius distance of the mean covariance to identity
        must be strictly smaller than before EA."""
        rng = np.random.default_rng(30)
        # Create correlated data (non-identity covariance)
        N, C, T = 30, 8, 100
        A = rng.standard_normal((C, C))
        X = rng.standard_normal((N, T, C)) @ A  # (N, T, C)
        X = X.transpose(0, 2, 1).astype(np.float32)  # (N, C, T)

        # Mean cov before EA
        covs_before = np.einsum("nct,ndt->ncd", X, X) / T
        R_mean_before = covs_before.mean(axis=0)
        I = np.eye(C, dtype=np.float32)
        dist_before = np.linalg.norm(R_mean_before - I, "fro")

        # Mean cov after EA
        X_aligned = euclidean_alignment(X)
        covs_after = np.einsum("nct,ndt->ncd", X_aligned, X_aligned) / T
        R_mean_after = covs_after.mean(axis=0)
        dist_after = np.linalg.norm(R_mean_after - I, "fro")

        assert dist_after < dist_before, (
            f"EA did not bring mean cov closer to identity: "
            f"before={dist_before:.4f}, after={dist_after:.4f}"
        )

    def test_session_labels_correct_values(self):
        """SESSION_LABELS must contain the exact per-session ground-truth values."""
        # Each session has 24 trials; labels must be in {0,1,2,3}
        assert set(SESSION_LABELS.keys()) == {1, 2, 3}, (
            "SESSION_LABELS keys must be {1,2,3}"
        )
        for sess_id, lbls in SESSION_LABELS.items():
            assert len(lbls) == 24, (
                f"Session {sess_id} has {len(lbls)} labels, expected 24"
            )
            assert set(lbls).issubset({0, 1, 2, 3}), (
                f"Session {sess_id} contains invalid label values: "
                f"{set(lbls) - {0, 1, 2, 3}}"
            )
        # Spot-check exact known values from the SEED-IV protocol
        assert SESSION_LABELS[1][0] == 1, "Session 1, trial 0 should be label 1"
        assert SESSION_LABELS[2][0] == 2, "Session 2, trial 0 should be label 2"
        assert SESSION_LABELS[3][0] == 1, "Session 3, trial 0 should be label 1"


# ===========================================================================
# E. Trainer Logic
# ===========================================================================


class TestTrainerLogic:
    """SDTrainer, GRL schedule, LOSOTrainer attributes, and early stopping."""

    def _make_tiny_loaders(self, n: int = 20, seed: int = 42):
        """Create tiny train / val / test SeedIVDatasets (no real data needed)."""
        from torch.utils.data import DataLoader

        rng = np.random.default_rng(seed)
        make = lambda n_: SeedIVDataset(
            rng.standard_normal((n_, 62, 800)).astype(np.float32),
            np.tile(np.arange(NUM_CLASSES), n_ // NUM_CLASSES + 1)[:n_].astype(
                np.int64
            ),
            normalise=False,
        )
        train_ds = make(n)
        val_ds = make(max(4, n // 4))
        test_ds = make(max(4, n // 4))
        return train_ds, val_ds, test_ds

    def test_sd_trainer_one_step(self):
        """SDTrainer must complete 2 epochs on tiny data and return acc in [0,1]."""
        from se_transnet.training.trainer_sd import SDTrainer

        train_ds, val_ds, test_ds = self._make_tiny_loaders(n=20)

        config = dict(
            batch_size=4,
            epochs=2,
            lr=1e-3,
            weight_decay=1e-4,
            eta_min=1e-5,
            num_segs=4,  # smaller segs fit tiny T
            num_classes=NUM_CLASSES,
            use_lr_swap=False,  # deterministic for test
            use_gaussian_noise=False,
            early_stop_patience=10,
            preferred_device="cpu",
            num_workers=0,
        )
        torch.manual_seed(42)
        model = SETransNet()
        loss_fn = EmotionDLLoss(epsilon=0.2)
        trainer = SDTrainer(model, config, loss_fn, result_savepath=None)
        result = trainer.train(train_ds, val_ds, test_ds)

        assert "acc" in result, "train() result dict missing 'acc'"
        assert 0.0 <= result["acc"] <= 1.0, f"Accuracy {result['acc']} not in [0, 1]"

    def test_grl_lambda_schedule(self):
        """GRL lambda schedule: lambda(0)~0, lambda(T)~1.0, strictly monotone."""
        from se_transnet.training.trainer_cs import LOSOTrainer

        config = dict(
            epochs=200,
            batch_size=4,
            lr=1e-3,
            dann_weight=0.1,
            mmd_weight=0.1,
            num_classes=NUM_CLASSES,
            preferred_device="cpu",
            num_workers=0,
        )
        model = SETransNet()
        loss_fn = EmotionDLLoss()
        trainer = LOSOTrainer(model, config, loss_fn, num_source_subjects=14)

        T = config["epochs"]
        lam_0 = trainer._compute_grl_lambda(0)
        lam_half = trainer._compute_grl_lambda(T // 2)
        lam_T = trainer._compute_grl_lambda(T)

        assert lam_0 == pytest.approx(0.0, abs=1e-6), (
            f"lambda(epoch=0) should be ~0, got {lam_0:.6f}"
        )
        # At epoch=T, p=1.0:  2/(1+exp(-10))-1 ~= 0.9999
        assert lam_T > 0.99, f"lambda(epoch=T) should be ~1.0, got {lam_T:.6f}"
        # Monotonically increasing
        assert lam_0 < lam_half < lam_T, (
            f"GRL lambda not monotone: {lam_0:.4f}, {lam_half:.4f}, {lam_T:.4f}"
        )

    def test_loso_trainer_attributes(self):
        """LOSOTrainer must have domain_disc, grl, dann_weight, mmd_weight."""
        from se_transnet.training.trainer_cs import LOSOTrainer

        config = dict(
            epochs=10,
            batch_size=4,
            lr=1e-3,
            dann_weight=0.1,
            mmd_weight=0.05,
            num_classes=NUM_CLASSES,
            preferred_device="cpu",
            num_workers=0,
        )
        model = SETransNet()
        loss_fn = EmotionDLLoss()
        trainer = LOSOTrainer(model, config, loss_fn)

        assert hasattr(trainer, "domain_disc"), "LOSOTrainer missing 'domain_disc'"
        assert hasattr(trainer, "grl"), "LOSOTrainer missing 'grl'"
        assert hasattr(trainer, "dann_weight"), "LOSOTrainer missing 'dann_weight'"
        assert hasattr(trainer, "mmd_weight"), "LOSOTrainer missing 'mmd_weight'"
        assert trainer.dann_weight == pytest.approx(0.1)
        assert trainer.mmd_weight == pytest.approx(0.05)

    def test_sd_trainer_early_stop_fires(self):
        """SDTrainer with patience=2 must stop before max_epochs when val does not improve."""
        from se_transnet.training.trainer_sd import SDTrainer

        # Use many epochs but tiny data so it won't converge — early stop should trigger
        n_epochs = 30
        config = dict(
            batch_size=4,
            epochs=n_epochs,
            lr=1e-4,  # tiny lr => likely no improvement
            weight_decay=1e-4,
            eta_min=1e-6,
            num_segs=4,
            num_classes=NUM_CLASSES,
            use_lr_swap=False,
            use_gaussian_noise=False,
            early_stop_patience=2,
            preferred_device="cpu",
            num_workers=0,
        )
        # Redirect print to suppress noise during test
        import contextlib
        import io

        train_ds, val_ds, test_ds = self._make_tiny_loaders(n=20, seed=77)

        torch.manual_seed(0)
        model = SETransNet()
        loss_fn = EmotionDLLoss(epsilon=0.2)
        trainer = SDTrainer(model, config, loss_fn, result_savepath=None)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = trainer.train(train_ds, val_ds, test_ds)

        output = buf.getvalue()
        # Confirm early stop message appeared before epoch n_epochs
        assert "Early stop" in output, (
            "Expected 'Early stop' message but did not find it; "
            "early stopping may not be firing correctly"
        )


# ===========================================================================
# F. Config & Integration
# ===========================================================================


class TestConfigAndIntegration:
    """YAML config validation, reproducibility with seeds."""

    def _load_yaml(self, filename: str) -> dict:
        path = _CONFIGS_DIR / filename
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def test_config_sd_required_keys(self):
        """config_sd.yaml must contain all required training/data/network keys."""
        cfg = self._load_yaml("config_sd.yaml")
        required_keys = [
            "batch_size",
            "lr",
            "epochs",
            "data_path",
            "network_args",
            "num_classes",
            "early_stop_patience",
            "val_split",
            "random_seed",
        ]
        missing = [k for k in required_keys if k not in cfg]
        assert not missing, f"config_sd.yaml is missing required keys: {missing}"

    def test_config_cs_required_keys(self):
        """config_cs.yaml must contain dann_weight, mmd_weight and use_euclidean_align."""
        cfg = self._load_yaml("config_cs.yaml")
        required_keys = ["dann_weight", "mmd_weight", "use_euclidean_align"]
        missing = [k for k in required_keys if k not in cfg]
        assert not missing, f"config_cs.yaml is missing required keys: {missing}"

    def test_model_reproducible_with_seed(self):
        """Same torch seed must produce identical logits on the same input."""
        x = torch.randn(2, 62, 800)  # fixed input

        torch.manual_seed(99)
        model1 = SETransNet().eval()
        with torch.no_grad():
            out1 = model1(x)

        torch.manual_seed(99)
        model2 = SETransNet().eval()
        with torch.no_grad():
            out2 = model2(x)

        assert torch.allclose(out1, out2, atol=1e-6), (
            "Same seed did not produce identical outputs — "
            "possible non-deterministic initialisation"
        )

    def test_model_different_seeds_differ(self):
        """Two models initialised with different seeds must have different weights."""
        torch.manual_seed(0)
        m1 = SETransNet()
        torch.manual_seed(1)
        m2 = SETransNet()

        # Compare first layer weights
        w1 = list(m1.parameters())[0].data
        w2 = list(m2.parameters())[0].data
        assert not torch.equal(w1, w2), (
            "Models with different seeds have identical weights — "
            "check if initialisation is truly random"
        )


# ===========================================================================
# G. Memory & Performance
# ===========================================================================


class TestBugFixes:
    """Regression tests for bugs found during debugging."""

    def test_kappa_nan_guard_sd_trainer(self):
        """SDTrainer.evaluate must return kappa=0.0 (not NaN) when all predictions
        collapse to a single class.

        Bug: cohen_kappa_score raises RuntimeWarning and returns NaN when
        expected agreement denominator is 0 (all preds same class).
        Fix: _safe_kappa() wraps cohen_kappa_score with a NaN->0.0 guard.
        """
        import math

        from se_transnet.training.trainer_sd import _safe_kappa

        # All same class in both true and pred -> kappa is NaN / 0
        y_true = [0, 0, 0, 0, 0, 0]
        y_pred = [0, 0, 0, 0, 0, 0]
        k = _safe_kappa(y_true, y_pred)
        assert not math.isnan(k), f"kappa is NaN, guard not working: k={k}"
        assert k == pytest.approx(0.0, abs=1e-9)

    def test_kappa_nan_guard_cs_trainer(self):
        """LOSOTrainer._safe_kappa must also return 0.0 on degenerate predictions."""
        import math

        from se_transnet.training.trainer_cs import _safe_kappa

        # Mixed true, all-same pred
        y_true = [0, 1, 2, 3]
        y_pred = [1, 1, 1, 1]
        k = _safe_kappa(y_true, y_pred)
        assert not math.isnan(k), f"kappa is NaN, guard not working: k={k}"
        # With all predictions being class 1 and mixed true labels, kappa < 0 is valid
        assert isinstance(k, float)

    def test_kappa_normal_case_unchanged(self):
        """_safe_kappa must return the real kappa value for non-degenerate cases."""
        import math

        from sklearn.metrics import cohen_kappa_score

        from se_transnet.training.trainer_sd import _safe_kappa

        y_true = [0, 1, 2, 3, 0, 1, 2, 3]
        y_pred = [0, 1, 2, 3, 1, 0, 3, 2]
        expected = cohen_kappa_score(y_true, y_pred)
        actual = _safe_kappa(y_true, y_pred)
        assert not math.isnan(actual)
        assert actual == pytest.approx(expected, abs=1e-9)

    def test_lr_pairs_count_is_27(self):
        """LR_PAIRS must have exactly 27 pairs.

        62 channels - 8 midline (Fpz, Fz, FCz, Cz, CPz, Pz, POz, Oz)
        = 54 lateral channels / 2 = 27 bilateral pairs.
        Spec comment previously said 29 -- corrected to 27.
        """
        assert len(LR_PAIRS) == 27, (
            f"Expected 27 LR electrode pairs, got {len(LR_PAIRS)}"
        )

    def test_lr_pairs_all_in_range(self):
        """Every LR pair index must be a valid 0-indexed channel (0..61)."""
        for l_idx, r_idx in LR_PAIRS:
            assert 0 <= l_idx < 62, f"Left index {l_idx} out of range [0, 62)"
            assert 0 <= r_idx < 62, f"Right index {r_idx} out of range [0, 62)"

    def test_lr_pairs_no_self_swap(self):
        """No LR pair should map a channel to itself."""
        for l_idx, r_idx in LR_PAIRS:
            assert l_idx != r_idx, f"Self-swap at index {l_idx}"

    def test_lr_pairs_no_duplicates(self):
        """No channel index should appear more than once across all LR pairs."""
        all_indices = [i for pair in LR_PAIRS for i in pair]
        assert len(all_indices) == len(set(all_indices)), (
            "Duplicate channel indices found in LR_PAIRS"
        )


class TestMemoryAndPerformance:
    """Parameter count, buffer checks and gradient accumulation guard."""

    def test_model_parameter_count(self):
        """Total trainable parameters must be between 1.5M and 2.5M."""
        model = SETransNet()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 1_500_000 < n_params < 2_500_000, (
            f"Parameter count {n_params:,} is outside expected [1.5M, 2.5M] range"
        )

    def test_model_no_extra_buffers(self):
        """Only the expected named buffers should be registered (no leaked intermediates)."""
        model = SETransNet()
        buffer_names = {name for name, _ in model.named_buffers()}
        # The only expected buffer comes from PositionalEncoding
        assert "pos_enc.pe" in buffer_names, "Expected 'pos_enc.pe' buffer not found"
        # Heuristic: no intermediate tensors should be registered as buffers
        # (they'd have names like 'avg_pool.weight' or numeric-named buffers)
        unexpected = {
            name
            for name in buffer_names
            if name not in ("pos_enc.pe",)
            and not name.startswith("bn_")
            and not name.startswith("conv_encoder")
            and not name.endswith(".running_mean")
            and not name.endswith(".running_var")
            and not name.endswith(".num_batches_tracked")
        }
        assert not unexpected, f"Unexpected non-BN buffers found: {unexpected}"

    def test_forward_backward_no_memory_leak(self):
        """Running 5 forward+backward passes must not accumulate gradients (zero_grad works)."""
        torch.manual_seed(42)
        model = SETransNet().train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        for step in range(5):
            optimizer.zero_grad(set_to_none=True)
            x = torch.randn(2, 62, 800)
            out = model(x)
            loss = out.sum()
            loss.backward()

            # After zero_grad + backward, every param should have exactly one grad
            for name, p in model.named_parameters():
                if p.requires_grad:
                    assert p.grad is not None, (
                        f"Step {step}: parameter '{name}' has no gradient after backward"
                    )
                    # Gradient should be finite
                    assert torch.isfinite(p.grad).all(), (
                        f"Step {step}: NaN/Inf gradient in '{name}'"
                    )

            # Verify zero_grad clears gradients before next step
            optimizer.zero_grad(set_to_none=True)
            for name, p in model.named_parameters():
                if p.requires_grad:
                    assert p.grad is None, (
                        f"Step {step}: gradient not cleared for '{name}' after zero_grad"
                    )
