# SE-TransNet V4 (SEED-IV)

Reorganized implementation folder for SEED-IV EEG-only emotion recognition.

## Structure

- `configs/` training configs for SD and LOSO
- `data/` raw + preprocessed dataset files
- `docs/` project context (`ai-context.md`, `spec.md`, `index.md`, notes)
- `se_transnet/` core package (models, data loaders, training, metrics)
- `tests/` validation tests
- `train_sd.py` subject-dependent entrypoint
- `train_cs.py` LOSO/cross-subject entrypoint
- `preprocess_seed4.py` raw `.mat` to preprocessed `.npy`
- `quality_check.py` dataset quality checks

## Quick Run

```bash
python train_sd.py --config configs/config_sd.yaml
python train_cs.py --config configs/config_cs.yaml
```

Use `--data-path` to override dataset location.
