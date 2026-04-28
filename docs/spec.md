# SE-TransNet SEED-IV — Complete PRD
# Version: V4 | Supersedes V1 (50.5%), V2, V3

## 0. Targets
- SD: ≥88% mean accuracy (15 subjects, session-based split)
- LOSO: ≥75% mean accuracy (15 folds)
- Stretch: ≥82% CS (matching DFF-Net / SS-EMERGE)
- Baselines: RGNN 79.37% SD / 73.84% CS; PR-PL 83.55% SD / 74.41% CS

## 1. Dataset
- 15 subjects × 3 sessions × 24 trials × ~7 windows = ~504 windows/subject total
- 62 channels, 200Hz, 4s window = 800 samples
- Classes: 0=Neutral, 1=Sad, 2=Fear, 3=Happy (from ReadMe.txt)
- Session labels exactly:
  - session1 = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
  - session2 = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
  - session3 = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
- Pre-processed .npy: sub{N}_session{S}_data.npy [N,62,800], _label.npy [N]

## 2. SE-TransNet Architecture

### Full Shape Trace
```
Input: [B, 62, 800]
→ unsqueeze(1): [B, 1, 62, 800]
→ 5× TempConv + concat + BN2d(80): [B, 80, 62, 800]
→ DWConv2d(80,80,(62,1),groups=80): [B, 80, 1, 800]
→ PWConv2d(80,80,(1,1)): [B, 80, 1, 800]
→ BN2d + ELU + Dropout(0.25): [B, 80, 1, 800]
→ squeeze(dim=2): [B, 80, 800]
→ AvgPool1d(40,10): [B, 80, 77]
→ VarPool(40,10): [B, 80, 77]
→ rearrange 'b d n -> b n d': [B, 77, 80] × 2
→ + PositionalEncoding(77,80): [B, 77, 80] × 2
→ 6× SharedTransformerEnc(h=8): [B, 77, 80] × 2
→ unsqueeze + concat dim=2: [B, 77, 2, 80]
→ Conv2d(77,64,(2,1)) + BN + ELU: [B, 64, 1, 80]
→ Conv2d(64,64,(1,1)) + BN + ELU + Drop: [B, 64, 1, 80]
→ flatten: [B, 5120]
→ FC1(5120,256) + BN + ELU + Drop(0.5): [B, 256]
→ FC2(256,64) + ELU + Drop(0.3): [B, 64]
→ FC3(64,4): [B, 4]
```

### Temporal Filter Bank (5 branches, F1=16)
```
Branch 1: Conv2d(1,16,(1,25),  pad=(0,12))  → 125ms @ 200Hz
Branch 2: Conv2d(1,16,(1,51),  pad=(0,25))  → 255ms
Branch 3: Conv2d(1,16,(1,101), pad=(0,50))  → 505ms
Branch 4: Conv2d(1,16,(1,201), pad=(0,100)) → 1005ms
Branch 5: Conv2d(1,16,(1,401), pad=(0,200)) → 2005ms
```
Why: Emotion needs slow dynamics (delta 1-4Hz → 250-1000ms, theta 4-8Hz → 125-250ms).
Original [15,25,51,65] captured only 60-260ms — designed for fast MI responses.

### Spatial Processing (Depthwise Separable)
```
DWConv2d(80,80,(62,1),groups=80)  → 80×62 params (vs 80×80×62 naive)
PWConv2d(80,80,(1,1))             → 80×80 params
BN2d(80) + ELU + Dropout(0.25)
```

### VarPool2D (vectorized)
```python
windows = x.unfold(-1, 40, 10)              # [B,80,77,40]
log_var = torch.log(torch.clamp(
            windows.float().var(-1), 1e-6, 1e6)).to(x.dtype)  # [B,80,77]
```

### Positional Encoding (sinusoidal, fixed)
```
PE[pos, 2i]   = sin(pos / 10000^(2i/80))
PE[pos, 2i+1] = cos(pos / 10000^(2i/80))
for pos in 0..76, i in 0..39
```

### Self-Attention (shared, N=6, h=8, d_m=80)
```
Pre-LN → MHA(d_m=80, h=8, d_k=10, attn_drop=0.1) → residual
Pre-LN → FFN(80→320→80, GELU, drop=0.1) → residual
× 6 layers, shared weights between avg and var streams
```

## 3. Training Hyperparameters

### Subject-Dependent
```yaml
batch_size: 32
epochs: 200
lr: 1e-3
weight_decay: 1e-4
patience: 30
num_segs: 8
label_smoothing: 0.05
emotion_dl_epsilon: 0.2
optimizer: AdamW
scheduler: CosineAnnealingLR(T_max=200, eta_min=1e-5)
```

### Cross-Subject (LOSO)
```yaml
batch_size: 128
epochs: 200
lr: 1e-3
weight_decay: 1e-4
patience: 40
dann_weight: 0.1
mmd_weight: 0.1
use_euclidean_align: true
grl_lambda: 2/(1+exp(-10*p))-1
```

## 4. Loss Functions

### EmotionDLLoss (ε=0.2)
Soft label matrix encoding valence-arousal distances:
```
         Neut   Sad    Fear   Happy
Neutral [0.80,  0.10,  0.05,  0.05]   # close to Sad
Sad     [0.10,  0.80,  0.05,  0.05]   # close to Neutral
Fear    [0.05,  0.05,  0.80,  0.10]   # close to Happy (high arousal)
Happy   [0.05,  0.05,  0.10,  0.80]   # close to Fear (high arousal)
```

## 5. Data Split Protocols

### SD — Session-Based (CORRECT — fixes V1 leakage)
- Train: sessions 1+2 (all windows from both sessions)
- Test: session 3 (all windows)
- Validation: 15% of training data (stratified, from sessions 1+2 only)
- NEVER randomly split across sessions

### LOSO — Leave-One-Subject-Out
- For each fold k: train on 14 subjects, test on subject k
- Apply Euclidean Alignment per-subject BEFORE pooling
- Validation: 15% of training pool

## 6. Augmentation
- SSR (Signal Segmentation & Recombination): Ns=8 segments, same-class mixing
- LR hemisphere swap (p=0.5): swap 27 electrode pairs (verified: 62ch - 8 midline = 54 lateral / 2 = 27 pairs)
- Gaussian noise (p=0.3, σ=0.01×std)
- Channel dropout (p=0.1): zero random channels

## 7. Acceptance Criteria
- SD mean accuracy ≥88% across 15 subjects
- LOSO mean accuracy ≥75% across 15 folds
- No data leakage (session-based split enforced)
- No class collapse (all 4 classes detected)
- Reproducible: seed=42, deterministic CUDNN
- Runs on Kaggle P100/T4 without OOM
