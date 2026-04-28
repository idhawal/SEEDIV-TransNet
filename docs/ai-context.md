# AI Context — SE-TransNet V4 Coding Rules

## Project
4-class EEG emotion recognition on SEED-IV (62ch, 200Hz).
Adapting EEG-TransNet (Motor Imagery) → emotion classification.
Platform: Kaggle Notebook (GPU P100/T4), PyTorch, modular .py files.
Version: V4 — target ≥88% SD, ≥75% LOSO.

## Critical Constants (NEVER CHANGE)
```python
NUM_CHANNELS     = 62       # SEED-IV ESI NeuroScan 62-channel cap
SAMPLING_RATE    = 200      # Hz — SEED-IV preprocessed rate
WINDOW_SECONDS   = 4        # seconds per segment
NUM_SAMPLES      = 800      # 4s × 200Hz
NUM_CLASSES      = 4        # Neutral, Sad, Fear, Happy
NUM_SUBJECTS     = 15
NUM_SESSIONS     = 3
TRIALS_PER_SESSION = 24
LABEL_MAP        = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}

# Architecture-derived constants
EMBED_DIM        = 80       # 5 branches × F1=16
T_SEQ            = 77       # (800-40)//10 + 1
POOL_SIZE        = 40       # NOT 50 (250Hz value)
POOL_STRIDE      = 10       # NOT 15 (250Hz value)
TEMPORAL_KERNELS = [25, 51, 101, 201, 401]  # 125/255/505/1005/2005ms @200Hz
SA_DEPTH         = 6
SA_HEADS         = 8        # 80/8=10 per head
ATTN_DROP        = 0.1      # NOT 0.5
```

## Session Labels (from ReadMe.txt — AUTHORITATIVE)
```python
SESSION_LABELS = {
    1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}
```

## Code Style
- Python 3.10+, PyTorch 2.0+, einops, scipy, sklearn, mne
- f-strings for all prints; Google-style docstrings on all public classes/functions
- set_seed(42) before every build_net() and every fold
- Max line length: 100
- Use pathlib.Path for file paths, not raw strings
- Use np.float32 for EEG data; np.int64 for labels

## NEVER DO

### Data
- ❌ NEVER shuffle windows across sessions before split → data leakage (V1 root cause)
- ❌ NEVER compute normalization stats on test data
- ❌ NEVER treat labels as 1-indexed (ReadMe uses 0=Neutral already)
- ❌ NEVER apply EA before the train/test split
- ❌ NEVER use overlapping windows in the test set
- ❌ NEVER apply per-trial normalization if per-window normalization is enabled

### Model
- ❌ NEVER use Python for-loop in VarPool2D.forward() → use x.unfold()
- ❌ NEVER use kernels [15,25,51,65] or [12,20,40,52] — too short for emotion
- ❌ NEVER use pool_size=50, pool_stride=15 — those are 250Hz MI values
- ❌ NEVER use single FC layer for classification
- ❌ NEVER use embed_dim=128 with 5 branches (misaligned; use 80=5×16)
- ❌ NEVER use embed_dim=32 (MI value, too small for 62 channels)
- ❌ NEVER use squeeze() without specifying dim — batch_size=1 breaks
- ❌ NEVER use variable name `input` — shadows builtin

### Training
- ❌ NEVER use batch_size > 64 for SD (only ~300 train samples per subject)
- ❌ NEVER use torch.cuda.amp.GradScaler() → use torch.amp.GradScaler('cuda')
- ❌ NEVER use torch.cuda.amp.autocast() → use torch.amp.autocast('cuda')
- ❌ NEVER use Adam without weight_decay → use AdamW
- ❌ NEVER import visdom (not on Kaggle)
- ❌ NEVER put EmotionDLLoss soft_labels on CPU while model on GPU (V3 crash)
- ❌ NEVER use bare nn.Dropout() → always specify p explicitly
- ❌ NEVER use dropout > 0.5 in attention — kills discriminative features

## ALWAYS DO
- ✅ Always cast data to .float() before model
- ✅ Always use non_blocking=True with .to(device) when using pinned memory
- ✅ Always clip gradients to max_norm=1.0
- ✅ Always use class-weighted loss (SEED-IV has class imbalance)
- ✅ Always report F1-macro alongside accuracy
- ✅ Always use Euclidean Alignment per-subject for cross-subject
- ✅ Always use drop_last=True in training DataLoader
- ✅ Always save best model by validation accuracy, not training accuracy
- ✅ Always split at session level for SD (train s1+s2, test s3)

## Key Shape Trace
```
[B,62,800] → [B,1,62,800] → [B,80,62,800] → [B,80,1,800] →
  avg:[B,80,77]  var:[B,80,77] →
  [B,77,80] × 2 → (+PE) → SA(N=6) →
  [B,77,2,80] → ConvEnc → [B,64,1,80] → flatten → [B,5120] →
  [B,256] → [B,64] → [B,4]
```

## Benchmarks
SD: RGNN 79.37%, PR-PL 83.55%, SS-EMERGE ~85%
CS: DANN 54.63%, RGNN 73.84%, PR-PL 74.41%, DFF-Net 82.32%, SS-EMERGE 81.51%
V1 baseline: SD 50.50%, CS 39.42% (to beat)

## Key Constant Values
num_samples = 800       # 4s × 200Hz
num_channels = 62
T_seq = 77              # (800-40)//10+1

## Preprocessing
- Bandpass: 0.5–75 Hz
- Notch: 50 Hz (Q=30)
- Normalization: per-window per-channel z-score
- Windowing: 4s windows with 50% overlap (stride=2s)
