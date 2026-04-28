# Index -- SE-TransNet V4 Documentation Map

## Key Context Files

| File | Purpose |
|------|---------|
| [ai-context.md](./ai-context.md) | Coding rules, constants, NEVER-DO list |
| [spec.md](./spec.md) | Full PRD with architecture spec |
| [Blueprint](../../SE_TransNet_SEED_IV_Blueprint.docx) | 12-section technical spec from Claude |
| [ReadMe.txt](../../ReadMe.txt) | AUTHORITATIVE session labels |

## V4 Implementation Files

| File | LOC | Status |
|------|-----|--------|
| configs/config_sd.yaml | 45 | Done |
| configs/config_cs.yaml | 45 | Done |
| se_transnet/data/datasets.py | 230 | Done |
| se_transnet/data/augmentation.py | 190 | Done |
| se_transnet/models/modules.py | 260 | Done |
| se_transnet/models/se_transnet.py | 250 | Done |
| se_transnet/training/losses.py | 200 | Done |
| se_transnet/training/trainer_sd.py | 280 | Done |
| se_transnet/evaluation/metrics.py | 170 | Done |
| train_sd.py | 180 | Done |
| train_cs.py | 195 | Done |
| test_shapes.py | 120 | Done - run requires working torch install |

## Architecture: V3 vs V4 Changes

| Parameter | V3 (crashed) | V4 (current) |
|-----------|-------------|--------------|
| Kernels | [12,20,40,52] 4-branch | [25,51,101,201,401] 5-branch |
| embed_dim | 128 | 80 (5x16) |
| pool_stride | 12 | 10 (T_seq=77) |
| Spatial conv | Naive Conv2d | Depthwise separable |
| Positional enc | None | Sinusoidal |
| attn_drop | 0.5 | 0.1 |
| FC head | Single linear | 3-layer (5120->256->64->4) |
| ConvEncoder | 1-layer | 2-layer |
| Data split | Random % (LEAKAGE) | Session-based (s1+s2 train, s3 test) |
| Batch size SD | 128 | 32 |
| Total params | 2,210,823 | 1,833,588 |
| Loss device bug | CRASH | Fixed (register_buffer) |

## Key Shape Trace
```
[B,62,800] -> [B,1,62,800] -> [B,80,62,800] -> [B,80,1,800] ->
  avg:[B,80,77]  var:[B,80,77] ->
  [B,77,80] x2 -> (+PE) -> SA(N=6) ->
  [B,77,2,80] -> ConvEnc -> [B,64,1,80] -> flatten -> [B,5120] ->
  [B,256] -> [B,64] -> [B,4]
```

## Labels (from ReadMe.txt)
```
0=Neutral, 1=Sad, 2=Fear, 3=Happy
s1=[1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
s2=[2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
s3=[1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
```

## How to Run

### Subject-Dependent (local)
```bash
py -3.9 train_sd.py --data-path /path/to/npy --subjects 1
```

### LOSO Cross-Subject (local)
```bash
py -3.9 train_cs.py --data-path /path/to/npy --subjects 1
```

### Kaggle (recommended for GPU)
Upload the se_transnet/ package as a Kaggle dataset, then import in notebook.

## Jupyter Notebooks

| File | Purpose |
|------|---------|
| (V3)seed-iv-eeg-transnet-adaptation.ipynb | Latest code — use as base |

