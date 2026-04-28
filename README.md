# SE-TransNet (SEED-IV)

A PyTorch implementation of **SE-TransNet** — an emotion-adapted EEG Transformer for 4-class emotion recognition on the **SEED-IV** dataset (Neutral, Sad, Fear, Happy). The model fuses a multi-scale temporal convolution bank, depthwise-separable spatial filtering, dual (mean + log-variance) token streams, and a shared 6-layer self-attention encoder with sinusoidal positional encoding.

> Targets: **≥88%** subject-dependent (SD) accuracy and **≥75%** leave-one-subject-out (LOSO) cross-subject accuracy on SEED-IV, with stretch parity to DFF-Net / SS-EMERGE.

---

## ✨ Highlights

- **5-branch temporal filter bank** with kernels `[25, 51, 101, 201, 401]` @ 200 Hz (125 ms – 2 s receptive fields) tuned for slow emotion dynamics in delta/theta bands.
- **Depthwise-separable spatial conv** over 62 EEG channels (parameter-efficient vs. naive `Conv2d`).
- **Dual-stream tokens**: mean-pooled and log-variance-pooled features, both passed through a **shared** 6-layer Transformer encoder (`d_model=80`, `h=8`).
- **Sinusoidal positional encoding** on a 77-token sequence (200 ms windows, 50 ms hop).
- **EmotionDLLoss**: soft label distribution encoding valence–arousal proximity (ε=0.2).
- **Domain adaptation for LOSO**: DANN (gradient reversal) + MMD + Euclidean alignment.
- **Augmentations**: segment-level recombination, left/right channel swap, Gaussian noise.
- **Strict, leakage-free protocols**: SD uses session-based split (train: s1+s2, test: s3); CS uses LOSO across 14 source subjects.

---

## 📁 Repository Structure

```
.
├── configs/
│   ├── config_sd.yaml            # Subject-dependent training config
│   └── config_cs.yaml            # LOSO cross-subject training config
├── docs/
│   ├── spec.md                   # Full PRD / architecture spec
│   ├── ai-context.md             # Coding rules & invariants
│   ├── index.md                  # Documentation map
│   └── SEED-IV_subject_info.txt  # Authoritative session labels
├── se_transnet/                  # Core package
│   ├── data/
│   │   ├── datasets.py           # SEED-IV dataset + loaders
│   │   └── augmentation.py       # Segment recomb, LR-swap, noise
│   ├── models/
│   │   ├── modules.py            # PE, VarPool2D, TransformerEncoder
│   │   └── se_transnet.py        # Main SETransNet model
│   ├── training/
│   │   ├── losses.py             # EmotionDLLoss, weighted CE, DANN/MMD
│   │   ├── trainer_sd.py         # SD training loop
│   │   └── trainer_cs.py         # LOSO training loop
│   └── evaluation/
│       └── metrics.py            # Accuracy, F1, confusion matrix
├── tests/
│   └── test_suite.py             # Validation tests
├── preprocess_seed4.py           # Raw .mat → preprocessed .npy
├── quality_check.py              # Dataset sanity checks
├── test_shapes.py                # Forward-pass shape verification
├── train_sd.py                   # Subject-dependent entrypoint
└── train_cs.py                   # LOSO/cross-subject entrypoint
```

---

## 🧠 Architecture at a Glance

Full shape trace (batch `B`):

```
Input:                       [B, 62, 800]              # 62 channels × 4 s @ 200 Hz
unsqueeze:                   [B, 1, 62, 800]
5× TempConv + concat + BN:   [B, 80, 62, 800]          # embed_dim = 5 × F1(16)
DWConv(62,1) + PWConv(1,1):  [B, 80, 1, 800]           # depthwise-separable spatial
ELU + Dropout(0.25), squeeze:[B, 80, 800]
AvgPool1d(40, 10):           [B, 80, 77]               # mean stream
VarPool1d(40, 10):           [B, 80, 77]               # log-variance stream
rearrange + PosEnc:          [B, 77, 80] × 2
Shared Transformer × 6:      [B, 77, 80] × 2           # h=8, d_k=10, FFN ratio 4
concat streams:              [B, 77, 2, 80]
ConvEncoder (2 layers):      [B, 64, 1, 80]
flatten:                     [B, 5120]
FC head 5120→256→64→4:       [B, 4]                    # logits
```

Total parameters: **~1.83M**.

See [`docs/spec.md`](docs/spec.md) for the complete PRD, hyperparameter rationale, and design history (V1 → V4).

---

## 🧪 Dataset

**SEED-IV** — 15 subjects × 3 sessions × 24 trials, recorded at 200 Hz on a 62-channel ESI NeuroScan cap.

- Window: 4 s (800 samples), 50% overlap (stride 2 s)
- Classes: `0=Neutral, 1=Sad, 2=Fear, 3=Happy`
- Session label vectors are defined verbatim in [`docs/spec.md`](docs/spec.md).

> SEED-IV is distributed by the BCMI Lab at SJTU and requires a signed academic license. Obtain the raw `.mat` files from the official source: <https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html>.

---

## 🚀 Getting Started

### 1. Environment

Tested with **Python 3.9+** and **PyTorch 2.x** (CUDA recommended).

```bash
# Create environment
python -m venv .venv && source .venv/bin/activate   # or conda create -n setn python=3.9

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy einops pyyaml scikit-learn tqdm
```

### 2. Preprocess the raw SEED-IV `.mat` files

```bash
python preprocess_seed4.py --raw-path /path/to/SEED-IV/eeg_raw_data \
                           --save-path ./data/seed4_preprocessed
```

The pipeline applies a 0.5–75 Hz bandpass + 50 Hz notch filter, performs 4 s / 50%-overlap windowing, rejects high-amplitude artifacts, and writes per-subject-session `sub{N}_session{S}_{data,label}.npy` arrays.

Optional flags:
- `--inspect` &nbsp; print metadata for the discovered files
- `--no-filter` &nbsp; skip filtering (use only on already-filtered data)

### 3. Sanity checks

```bash
python quality_check.py --data-path ./data/seed4_preprocessed
python test_shapes.py                # verify model forward pass
pytest tests/                        # run validation suite
```

### 4. Train

**Subject-dependent (per-subject, train: s1+s2, test: s3):**

```bash
python train_sd.py --config configs/config_sd.yaml \
                   --data-path ./data/seed4_preprocessed \
                   --subjects 1
```

**Leave-One-Subject-Out (cross-subject):**

```bash
python train_cs.py --config configs/config_cs.yaml \
                   --data-path ./data/seed4_preprocessed
```

Both entrypoints accept `--data-path` to override the dataset location and write checkpoints / logs under `out_folder` (`./output/SD` or `./output/CS` by default).

---

## ⚙️ Key Hyperparameters

| Setting              | SD               | LOSO (CS)        |
|----------------------|------------------|------------------|
| Batch size           | 32               | 128              |
| Epochs / patience    | 200 / 30         | 200 / 40         |
| Optimizer            | AdamW (lr=1e-3, wd=1e-4) | AdamW (lr=1e-3, wd=1e-4) |
| Scheduler            | CosineAnnealing  | CosineAnnealing  |
| Label smoothing      | 0.05             | 0.05             |
| EmotionDL ε          | 0.2              | 0.2              |
| Domain losses        | —                | DANN 0.1 + MMD 0.1 + Euclidean align |
| Augmentation         | LR-swap, segment recomb, Gaussian noise | + |

Full configs live in [`configs/config_sd.yaml`](configs/config_sd.yaml) and [`configs/config_cs.yaml`](configs/config_cs.yaml).

---

## 📊 Results (Targets)

| Protocol | This work (target) | RGNN | PR-PL | DFF-Net |
|----------|--------------------|------|-------|---------|
| SD       | **≥88%**           | 79.37% | 83.55% | — |
| LOSO/CS  | **≥75%** (stretch ≥82%) | 73.84% | 74.41% | ~82% |

Reproduced numbers will be added once the full sweep completes — see `output/` after training for per-subject reports and confusion matrices.

---

## 🧰 Development Notes

- **Device handling**: all loss tensors use `register_buffer` to follow `model.to(device)` automatically.
- **Reproducibility**: `random_seed: 42` is set for NumPy, PyTorch, and CUDA.
- **No data leakage**: SD never randomly splits across sessions; LOSO never trains on the held-out subject.
- **Mixed precision** is supported by toggling AMP in the trainer.

---

## 📚 References

- Zheng, W.-L. et al. *EmotionMeter: A Multimodal Framework for Recognizing Human Emotions*, IEEE TCYB, 2018. (SEED-IV)
- Song, T. et al. *EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks* (RGNN baseline).
- PR-PL, DFF-Net, SS-EMERGE — see [`docs/spec.md`](docs/spec.md) for the full bibliography.

---

## 📝 License & Citation

This repository is released for academic and research use. If you build on this work, please cite the SEED-IV dataset paper and reference this repository.

```bibtex
@software{se_transnet_seediv,
  title  = {SE-TransNet: Emotion-Adapted EEG Transformer for SEED-IV},
  author = {Dhawal, I.},
  year   = {2025},
  url    = {https://github.com/idhawal/SEEDIV-TransNet}
}
```

---

## 🙋 Contributing

Issues and PRs are welcome — especially around:
- Additional baselines / ablation studies
- Improved domain-adaptation strategies for LOSO
- Reproducibility scripts (Docker, Kaggle notebooks)

Please run `pytest tests/` and `python test_shapes.py` before submitting.
