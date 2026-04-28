"""Comprehensive quality check for SE-TransNet V4."""
import sys, os, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

PASS, FAIL, WARN = 0, 0, 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1; print(f"  [PASS] {name}")
    else:
        FAIL += 1; print(f"  [FAIL] {name} -- {detail}")

def warn(name, detail=""):
    global WARN; WARN += 1; print(f"  [WARN] {name} -- {detail}")

def run_all():
    global PASS, FAIL, WARN
    print("=" * 65)
    print("SE-TransNet V4 -- COMPREHENSIVE QUALITY CHECK")
    print("=" * 65)

    # ── 1. Package Structure ──────────────────────────────────
    print("\n--- 1. Package Structure ---")
    base = Path(__file__).parent / "se_transnet"
    required = [
        "__init__.py", "data/__init__.py", "data/datasets.py",
        "data/augmentation.py", "models/__init__.py", "models/modules.py",
        "models/se_transnet.py", "training/__init__.py", "training/losses.py",
        "training/trainer_sd.py", "training/trainer_cs.py",
        "evaluation/__init__.py", "evaluation/metrics.py",
    ]
    for f in required:
        check(f"se_transnet/{f}", (base / f).exists(), "FILE MISSING")

    root = Path(__file__).parent
    for f in ["train_sd.py", "train_cs.py", "test_shapes.py",
              "preprocess_seed4.py", "configs/config_sd.yaml", "configs/config_cs.yaml"]:
        check(f, (root / f).exists(), "FILE MISSING")

    # ── 2. Imports ────────────────────────────────────────────
    print("\n--- 2. Module Imports ---")
    modules = {}
    for name, path in [
        ("datasets", "se_transnet.data.datasets"),
        ("augmentation", "se_transnet.data.augmentation"),
        ("modules", "se_transnet.models.modules"),
        ("se_transnet", "se_transnet.models.se_transnet"),
        ("losses", "se_transnet.training.losses"),
        ("trainer_sd", "se_transnet.training.trainer_sd"),
        ("trainer_cs", "se_transnet.training.trainer_cs"),
        ("metrics", "se_transnet.evaluation.metrics"),
    ]:
        try:
            modules[name] = __import__(path, fromlist=[name])
            check(f"import {path}", True)
        except Exception as e:
            check(f"import {path}", False, str(e))

    # ── 3. Model Architecture ─────────────────────────────────
    print("\n--- 3. Model Architecture ---")
    from se_transnet.models.se_transnet import SETransNet

    net = SETransNet(num_classes=4, num_samples=800, num_channels=62,
                     embed_dim=80, temporal_kernels=[25,51,101,201,401],
                     pool_size=40, pool_stride=10, num_heads=8,
                     fc_ratio=4, depth=6, attn_drop=0.1, fc_drop=0.5, spatial_drop=0.25)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    check("5-branch temporal bank", len(net.temporal_convs) == 5)
    kernels = [c.kernel_size[1] for c in net.temporal_convs]
    check("Kernel sizes [25,51,101,201,401]", kernels == [25,51,101,201,401], f"got {kernels}")
    check("embed_dim=80", net.bn_temporal.num_features == 80)
    check("Depthwise spatial (groups=80)", net.spatial_dw.groups == 80)
    check("Pointwise spatial (1x1)", net.spatial_pw.kernel_size == (1,1))
    check("6 transformer layers", len(net.transformer_encoders) == 6)
    check("Positional encoding exists", hasattr(net, 'pos_enc'))
    check("3-layer FC head", len(list(net.classifier.children())) >= 7,
          f"got {len(list(net.classifier.children()))} layers")
    check("Params ~1.8M", 1_500_000 < n_params < 2_500_000, f"got {n_params:,}")
    print(f"    Total trainable params: {n_params:,}")

    # ── 4. Forward Pass Shapes ────────────────────────────────
    print("\n--- 4. Forward Pass Shapes ---")
    x = torch.randn(4, 62, 800)
    out = net(x)
    check("Output shape (B,4)", out.shape == (4,4), f"got {out.shape}")

    T_seq = (800 - 40) // 10 + 1
    check("T_seq = 77", T_seq == 77, f"got {T_seq}")

    # Gradient flow
    out.sum().backward()
    n_grad = sum(1 for p in net.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in net.parameters() if p.requires_grad)
    check(f"All {n_total} params have gradients", n_grad == n_total, f"{n_grad}/{n_total}")

    # Feature extraction
    net.zero_grad()
    feats = net.extract_features(torch.randn(2, 62, 800))
    check("extract_features -> (B,256)", feats.shape == (2, 256), f"got {feats.shape}")

    # Batch size 1
    net.eval()
    with torch.no_grad():
        out1 = net(torch.randn(1, 62, 800))
    check("Batch size 1 works", out1.shape == (1, 4))

    # ── 5. Loss Functions ─────────────────────────────────────
    print("\n--- 5. Loss Functions ---")
    from se_transnet.training.losses import EmotionDLLoss, GradientReversalLayer, DomainDiscriminator, compute_mmd, compute_coral

    # EmotionDLLoss
    loss_fn = EmotionDLLoss(epsilon=0.2)
    logits = torch.randn(8, 4)
    targets = torch.tensor([0,1,2,3,0,1,2,3])
    loss_val = loss_fn(logits, targets)
    check("EmotionDLLoss computes", loss_val.item() > 0, f"loss={loss_val.item():.4f}")
    check("EmotionDLLoss soft_labels is buffer", 'soft_labels' in dict(loss_fn.named_buffers()))

    # With class weights
    weights = torch.tensor([1.2, 0.8, 1.5, 0.9])
    loss_w = EmotionDLLoss(epsilon=0.2, weight=weights)
    check("Weighted EmotionDL", loss_w(logits, targets).item() > 0)

    # Device safety: soft_labels should move with model
    if torch.cuda.is_available():
        loss_gpu = EmotionDLLoss(epsilon=0.2).cuda()
        check("EmotionDL GPU device safe", loss_gpu.soft_labels.device.type == 'cuda')

    # GRL
    grl = GradientReversalLayer(alpha=1.0)
    x_grl = torch.randn(4, 256, requires_grad=True)
    y_grl = grl(x_grl)
    check("GRL forward is identity", torch.allclose(x_grl, y_grl))
    grl.set_alpha(0.5)
    check("GRL alpha update", grl.alpha == 0.5)

    # Domain Discriminator
    dd = DomainDiscriminator(in_features=256, n_domains=14)
    dd_out = dd(torch.randn(4, 256))
    check("DomainDisc output (B,14)", dd_out.shape == (4, 14))

    # MMD
    src = torch.randn(16, 256)
    tgt = torch.randn(16, 256)
    mmd = compute_mmd(src, tgt)
    check("MMD computes", mmd.item() >= 0, f"mmd={mmd.item():.4f}")
    mmd_self = compute_mmd(src, src)
    check("MMD(x,x) ~ 0", mmd_self.item() < 0.1, f"mmd_self={mmd_self.item():.6f}")

    # CORAL
    coral = compute_coral(src, tgt)
    check("CORAL computes", coral.item() >= 0)

    # ── 6. Augmentation ───────────────────────────────────────
    print("\n--- 6. Augmentation ---")
    from se_transnet.data.augmentation import SSRAugmentor, lr_hemisphere_swap, gaussian_noise, channel_dropout

    ssr = SSRAugmentor(num_segs=8, num_classes=4, batch_size=32)
    fake_data = torch.randn(16, 62, 800)
    fake_labels = torch.tensor([0]*4 + [1]*4 + [2]*4 + [3]*4)
    aug_d, aug_l = ssr(fake_data, fake_labels)
    check("SSR produces augmented data", aug_d.numel() > 0)
    check("SSR shape (N,62,800)", aug_d.shape[1:] == (62, 800), f"got {aug_d.shape}")
    check("SSR labels valid", set(aug_l.numpy()).issubset({0,1,2,3}))

    swapped = lr_hemisphere_swap(fake_data, p=1.0)
    check("LR swap shape preserved", swapped.shape == fake_data.shape)

    noisy = gaussian_noise(fake_data, sigma_ratio=0.01, p=1.0)
    check("Gaussian noise shape preserved", noisy.shape == fake_data.shape)
    check("Noise adds variation", not torch.allclose(noisy, fake_data))

    dropped = channel_dropout(fake_data, p=0.5)
    check("Channel dropout shape", dropped.shape == fake_data.shape)
    check("Some channels zeroed", (dropped.abs().sum(dim=-1) == 0).any())

    # ── 7. Dataset & Loader ───────────────────────────────────
    print("\n--- 7. Dataset & Loader ---")
    from se_transnet.data.datasets import SeedIVDataset, SESSION_LABELS, euclidean_alignment, load_seediv_LOSO

    # Session labels
    check("Session 1 has 24 labels", len(SESSION_LABELS[1]) == 24)
    check("Session 2 has 24 labels", len(SESSION_LABELS[2]) == 24)
    check("Session 3 has 24 labels", len(SESSION_LABELS[3]) == 24)
    for s in [1, 2, 3]:
        check(f"Session {s} labels in {{0,1,2,3}}", set(SESSION_LABELS[s]).issubset({0,1,2,3}))

    # Dataset with synthetic data
    syn_data = np.random.randn(100, 62, 800).astype(np.float32)
    syn_labels = np.random.randint(0, 4, 100).astype(np.int64)
    syn_domains = np.random.randint(0, 14, 100).astype(np.int64)
    ds = SeedIVDataset(syn_data, syn_labels, domain_labels=syn_domains, normalise=True)
    check("Dataset length", len(ds) == 100)

    x_item, y_item, d_item = ds[0]
    check("Dataset domain label", isinstance(d_item, (int, np.integer)))

    # Real data checks (optional)
    data_root = Path(__file__).parent / "data" / "seed4_preprocessed"
    if data_root.exists():
        try:
            tr_d, tr_l, tr_dom, val_d, val_l, val_dom, te_d, te_l = load_seediv_LOSO(
                data_root, test_subject=1, config={"val_split": 0.1, "random_seed": 42}
            )
            check("LOSO domain labels length", len(tr_dom) == len(tr_l))
            if len(tr_dom) > 0:
                check("LOSO domain label range", tr_dom.min() >= 0 and tr_dom.max() <= 13)
        except Exception as e:
            warn("LOSO loader", f"{e}")

        # Overlap check (sampled) between session 1 and 3 for subject 1
        try:
            d1 = np.load(data_root / "sub1_session1_data.npy")
            d3 = np.load(data_root / "sub1_session3_data.npy")
            n = min(len(d1), len(d3), 200)
            def _hash_rows(arr):
                return {hash(arr[i].tobytes()) for i in range(n)}
            overlap = _hash_rows(d1) & _hash_rows(d3)
            check("No overlap S1 vs S3 (sample)", len(overlap) == 0, f"overlap={len(overlap)}")
        except Exception as e:
            warn("Overlap check", f"{e}")
    else:
        warn("Dataset path", f"missing: {data_root}")

    # ── 8. Trainers ───────────────────────────────────────────
    print("\n--- 8. Trainers ---")
    from se_transnet.training.trainer_sd import SDTrainer
    from se_transnet.training.trainer_cs import LOSOTrainer

    # SDTrainer attributes
    check("SDTrainer has train method", hasattr(SDTrainer, 'train'))
    check("SDTrainer has evaluate method", hasattr(SDTrainer, 'evaluate'))
    check("SDTrainer has train_one_epoch", hasattr(SDTrainer, 'train_one_epoch'))

    # LOSOTrainer attributes
    check("LOSOTrainer has train method", hasattr(LOSOTrainer, 'train'))
    check("LOSOTrainer has evaluate method", hasattr(LOSOTrainer, 'evaluate'))
    check("LOSOTrainer has train_one_epoch", hasattr(LOSOTrainer, 'train_one_epoch'))
    check("LOSOTrainer has _compute_grl_lambda", hasattr(LOSOTrainer, '_compute_grl_lambda'))

    # GRL lambda annealing
    config = {'preferred_device': 'cpu', 'batch_size': 8, 'epochs': 100,
              'lr': 1e-3, 'num_classes': 4, 'dann_weight': 0.1, 'mmd_weight': 0.1}
    small_net = SETransNet(num_classes=4, num_samples=800, num_channels=62,
                           embed_dim=80, pool_size=40, pool_stride=10, depth=2)
    small_loss = EmotionDLLoss(epsilon=0.2)
    trainer = LOSOTrainer(small_net, config, small_loss)
    lam_0 = trainer._compute_grl_lambda(0)
    lam_50 = trainer._compute_grl_lambda(50)
    lam_100 = trainer._compute_grl_lambda(100)
    check("GRL lambda(0) ~ 0", lam_0 < 0.1, f"got {lam_0:.4f}")
    check("GRL lambda(50) saturated (Ganin formula)", lam_50 > 0.9, f"got {lam_50:.4f}")
    check("GRL lambda(100) ~ 1", lam_100 > 0.9, f"got {lam_100:.4f}")

    # ── 9. Config Files ───────────────────────────────────────
    print("\n--- 9. Config Files ---")
    import yaml
    for cfg_name in ['configs/config_sd.yaml', 'configs/config_cs.yaml']:
        cfg_path = root / cfg_name
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            check(f"{cfg_name} parseable", cfg is not None)
            check(f"{cfg_name} has network_args", 'network_args' in cfg, str(list(cfg.keys())))
            if 'network_args' in cfg:
                na = cfg['network_args']
                check(f"{cfg_name} num_classes=4", na.get('num_classes') == 4)
                check(f"{cfg_name} num_samples=800", na.get('num_samples') == 800)
                check(f"{cfg_name} num_channels=62", na.get('num_channels') == 62)
                check(f"{cfg_name} embed_dim=80", na.get('embed_dim') == 80)
        else:
            check(f"{cfg_name} exists", False, "MISSING")

    # ── 10. Spec Compliance ───────────────────────────────────
    print("\n--- 10. Spec Compliance (V4 Requirements) ---")
    check("5-branch temporal (not 4)", len(net.temporal_convs) == 5)
    check("Depthwise separable (not naive)", net.spatial_dw.groups == 80)
    check("Sinusoidal PE present", hasattr(net, 'pos_enc'))
    check("Pre-norm transformer", hasattr(net.transformer_encoders[0], 'layernorm1'))
    check("3-layer FC (not single linear)", len(list(net.classifier.children())) > 3)
    check("EmotionDL uses register_buffer", 'soft_labels' in dict(EmotionDLLoss().named_buffers()))
    check("LOSOTrainer has DANN support", hasattr(trainer, 'domain_disc'))
    check("LOSOTrainer has MMD support", hasattr(trainer, 'use_mmd'))
    check("LOSOTrainer has GRL", hasattr(trainer, 'grl'))

    # ── 11. Data Leakage Check ────────────────────────────────
    print("\n--- 11. Data Leakage Prevention ---")
    # Verify load_seediv_SD uses session-based split
    import inspect
    from se_transnet.data.datasets import load_seediv_SD, load_seediv_LOSO
    sd_src = inspect.getsource(load_seediv_SD)
    check("SD: session-based split", 'train_sessions' in sd_src and 'test_sessions' in sd_src)
    check("SD: no random split", 'random_split' not in sd_src)

    loso_src = inspect.getsource(load_seediv_LOSO)
    check("LOSO: per-subject loop", 'test_subject' in loso_src)
    check("LOSO: EA integration", 'euclidean_alignment' in loso_src)

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    total = PASS + FAIL
    print(f"QUALITY CHECK COMPLETE: {PASS}/{total} passed, {FAIL} failed, {WARN} warnings")
    if FAIL == 0:
        print("STATUS: ALL CHECKS PASSED")
    else:
        print("STATUS: ISSUES FOUND - see [FAIL] items above")
    print("=" * 65)

if __name__ == '__main__':
    run_all()
