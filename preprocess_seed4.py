from __future__ import annotations

"""
Optimized SEED-IV Preprocessing Pipeline.

Purpose-built for emotion recognition with SE-TransNet V4.

Pipeline stages:
  1. Load raw EEG from .mat (62ch, 200Hz, keys: cz_eeg1..cz_eeg24)
  2. Bandpass filter 0.5-75 Hz (preserves delta/theta/alpha/beta/gamma)
  3. Notch filter 50 Hz (powerline interference removal)
  4. Per-window normalization (z-score per channel per window, downstream)
  5. 50% overlapping 4s windowing (stride=2s)
  6. Artifact rejection (windows with amplitude > threshold removed)
  7. Save per-subject-session .npy files

Key design decisions:
  - Bandpass 0.5-75 Hz: emotion dynamics are in delta(1-4), theta(4-8),
    alpha(8-13), beta(13-30), low-gamma(30-45), high-gamma(45-75). We keep all.
  - Per-window z-score (not per-trial, not global): prevents information
    leakage across trials while normalizing amplifier gain differences.
  - 50% overlap in windowing: improves data efficiency and smooths transitions.
  - Artifact rejection threshold: removes muscle/eye artifacts that would
    confuse the classifier.

Usage:
  python preprocess_seed4.py
  python preprocess_seed4.py --raw-path <path> --save-path <path>
  python preprocess_seed4.py --inspect
  python preprocess_seed4.py --no-filter  (skip filtering for pre-filtered data)
"""

import argparse
import glob
import os
import re
import sys
import time

import numpy as np
from scipy import signal as sig
from scipy.io import loadmat


# ================================================================
# Constants
# ================================================================

SESSION_LABELS = {
    1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}
EMOTION_MAP = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}
FS = 200          # Sampling rate (Hz)
WINDOW_SEC = 4    # Window duration (seconds)
WINDOW_SIZE = FS * WINDOW_SEC  # 800 samples
WINDOW_STRIDE = WINDOW_SIZE // 2  # 50% overlap
NUM_CHANNELS = 62
NUM_TRIALS = 24
NUM_SUBJECTS = 15
NUM_SESSIONS = 3

# Filter parameters
BANDPASS_LOW = 0.5    # Hz - removes DC drift
BANDPASS_HIGH = 75.0  # Hz - keep up to high-gamma
NOTCH_FREQ = 50.0     # Hz - powerline interference
NOTCH_Q = 30.0        # Quality factor for notch filter

# Artifact rejection
ARTIFACT_THRESHOLD = 200.0  # uV - windows with max|amplitude| > this are rejected
ARTIFACT_ENABLED = True


# ================================================================
# Signal Processing
# ================================================================

def design_bandpass(low: float, high: float, fs: float, order: int = 5):
    """Design Butterworth bandpass filter coefficients.

    Uses second-order sections (sos) for numerical stability,
    which is critical for narrow bands and high filter orders.
    """
    nyq = fs / 2.0
    low_norm = low / nyq
    high_norm = high / nyq
    sos = sig.butter(order, [low_norm, high_norm], btype='band', output='sos')
    return sos


def design_notch(freq: float, fs: float, Q: float = 30.0):
    """Design IIR notch filter to remove powerline interference."""
    b, a = sig.iirnotch(freq, Q, fs)
    return b, a


def apply_bandpass(data: np.ndarray, sos, axis: int = -1) -> np.ndarray:
    """Apply zero-phase bandpass filter using sosfiltfilt.

    Zero-phase filtering (forward + backward pass) prevents
    phase distortion which would corrupt temporal dynamics.
    """
    return sig.sosfiltfilt(sos, data, axis=axis).astype(np.float32)


def apply_notch(data: np.ndarray, b, a, axis: int = -1) -> np.ndarray:
    """Apply zero-phase notch filter."""
    return sig.filtfilt(b, a, data, axis=axis).astype(np.float32)


def per_trial_normalize(trial: np.ndarray) -> np.ndarray:
    """Optional per-trial z-score normalization (not used by default).

    Args:
        trial: (62, T) raw EEG for one trial.
    Returns:
        (62, T) normalized.
    """
    mean = trial.mean(axis=1, keepdims=True)
    std = trial.std(axis=1, keepdims=True)
    std = np.where(std < 1e-8, 1e-8, std)
    return ((trial - mean) / std).astype(np.float32)


def reject_artifacts(
    segments: np.ndarray,
    labels: np.ndarray,
    threshold: float = ARTIFACT_THRESHOLD,
) -> tuple:
    """Remove windows with extreme amplitude values.

    These are typically caused by eye blinks, jaw clenching,
    or electrode pops. Removing them prevents the model from
    learning artifact patterns instead of emotion patterns.
    """
    max_amp = np.abs(segments).max(axis=(1, 2))  # (N,)
    good_mask = max_amp < threshold
    n_rejected = (~good_mask).sum()
    if n_rejected > 0:
        print(f'    Artifact rejection: removed {n_rejected}/{len(segments)} '
              f'windows (threshold={threshold})')
    return segments[good_mask], labels[good_mask]


# ================================================================
# File handling
# ================================================================

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def find_trial_keys(mat_data: dict) -> list:
    """Find and sort trial keys, handling cz_eeg1..cz_eeg24 naming."""
    skip = {'__header__', '__version__', '__globals__'}
    candidates = [k for k in mat_data.keys() if k not in skip]

    # SEED-IV uses 'cz_eeg1' .. 'cz_eeg24'
    eeg_keys = [k for k in candidates if 'eeg' in k.lower()]
    if len(eeg_keys) >= NUM_TRIALS:
        eeg_keys.sort(key=natural_sort_key)
        return eeg_keys[:NUM_TRIALS]

    # Fallback
    candidates.sort(key=natural_sort_key)
    return candidates[:NUM_TRIALS]


def detect_raw_path(raw_path: str) -> str:
    """Auto-detect actual root containing session folders.

    Handles nested structures like:
      raw_path/eeg_raw_data/1/*.mat
      raw_path/1/*.mat
    """
    # Direct: raw_path/1/*.mat
    if glob.glob(os.path.join(raw_path, '1', '*.mat')):
        return raw_path

    # Nested: raw_path/eeg_raw_data/1/*.mat
    nested = os.path.join(raw_path, 'eeg_raw_data')
    if os.path.isdir(nested) and glob.glob(os.path.join(nested, '1', '*.mat')):
        return nested

    # Search one level deeper
    for d in os.listdir(raw_path):
        subdir = os.path.join(raw_path, d)
        if os.path.isdir(subdir) and glob.glob(os.path.join(subdir, '1', '*.mat')):
            return subdir

    return raw_path


def extract_subject_id(filename: str) -> int:
    """Extract subject ID from filenames like '1_20160518.mat'."""
    basename = os.path.basename(filename).replace('.mat', '')
    nums = re.findall(r'(\d+)', basename)
    if nums:
        return int(nums[0])
    raise ValueError(f'Cannot extract subject ID from {filename}')


# ================================================================
# Main processing
# ================================================================

def process_one_file(
    fpath: str,
    sub_id: int,
    sess_id: int,
    sos_bp,
    b_notch, a_notch,
    use_filter: bool = True,
    use_artifact_rejection: bool = True,
) -> tuple:
    """Process one .mat file -> (segments, labels).

    Pipeline per trial:
      1. Load raw (62, T)
      2. Bandpass 0.5-75 Hz
      3. Notch 50 Hz
      4. Segment into 4s windows with 50% overlap (stride=2s)
      5. Artifact rejection

    Returns:
        (segments: ndarray (N, 62, 800), labels: ndarray (N,))
    """
    mat = loadmat(fpath)
    trial_keys = find_trial_keys(mat)
    labels_list = SESSION_LABELS[sess_id]

    if len(trial_keys) < NUM_TRIALS:
        print(f'    [WARN] Expected {NUM_TRIALS} trials, found {len(trial_keys)}')

    all_segments = []
    all_labels = []
    total_duration = 0.0

    for trial_idx, key in enumerate(trial_keys):
        if trial_idx >= len(labels_list):
            break

        trial = mat[key]  # (62, T)
        if trial.ndim != 2 or trial.shape[0] != NUM_CHANNELS:
            print(f'    [SKIP] {key}: unexpected shape {trial.shape}')
            continue

        trial = trial.astype(np.float64)
        T = trial.shape[1]
        total_duration += T / FS

        # Stage 2-3: Filtering
        if use_filter:
            trial = apply_bandpass(trial, sos_bp, axis=1)
            trial = apply_notch(trial, b_notch, a_notch, axis=1)

        # Stage 4: Per-window normalization (downstream in dataset loader)
        # trial = per_trial_normalize(trial)

        # Stage 5: 50% overlapping 4s windowing
        n_windows = 1 + max(0, (T - WINDOW_SIZE) // WINDOW_STRIDE)
        label = labels_list[trial_idx]

        for w in range(n_windows):
            start = w * WINDOW_STRIDE
            end = start + WINDOW_SIZE
            seg = trial[:, start:end]
            if seg.shape[1] != WINDOW_SIZE:
                continue
            all_segments.append(seg)
            all_labels.append(label)

    if not all_segments:
        return np.array([]), np.array([])

    segments = np.array(all_segments, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    # Stage 6: Artifact rejection
    if use_artifact_rejection and ARTIFACT_ENABLED:
        segments, labels = reject_artifacts(segments, labels)

    return segments, labels


def process_all(
    raw_path: str,
    save_path: str,
    use_filter: bool = True,
    use_artifact_rejection: bool = True,
    inspect_only: bool = False,
) -> dict:
    """Process all subjects x sessions.

    Returns:
        stats dict: {(sub_id, sess_id): {'n_windows': int, 'duration_s': float}}
    """
    actual_root = detect_raw_path(raw_path)
    print(f'  Resolved raw path: {actual_root}')

    os.makedirs(save_path, exist_ok=True)

    # Pre-compute filter coefficients (once for all files)
    sos_bp = design_bandpass(BANDPASS_LOW, BANDPASS_HIGH, FS, order=5)
    b_notch, a_notch = design_notch(NOTCH_FREQ, FS, NOTCH_Q)

    stats = {}
    total_windows = 0
    t_global = time.time()

    for sess_id in range(1, NUM_SESSIONS + 1):
        sess_dir = os.path.join(actual_root, str(sess_id))
        if not os.path.isdir(sess_dir):
            print(f'  [SKIP] Session {sess_id} directory not found: {sess_dir}')
            continue

        mat_files = sorted(glob.glob(os.path.join(sess_dir, '*.mat')),
                           key=natural_sort_key)
        print(f'\n  Session {sess_id}: {len(mat_files)} subjects')

        for fpath in mat_files:
            sub_id = extract_subject_id(fpath)

            if inspect_only:
                mat = loadmat(fpath)
                skip = {'__header__', '__version__', '__globals__'}
                keys = [k for k in mat.keys() if k not in skip]
                print(f'    sub{sub_id}: {len(keys)} trials, '
                      f'first={mat[keys[0]].shape if keys else "N/A"}')
                continue

            t0 = time.time()
            print(f'  sub{sub_id:02d}_sess{sess_id}:', end=' ')

            segments, labels = process_one_file(
                fpath, sub_id, sess_id,
                sos_bp, b_notch, a_notch,
                use_filter=use_filter,
                use_artifact_rejection=use_artifact_rejection,
            )

            if len(segments) == 0:
                print('NO DATA')
                continue

            # Save
            dp = os.path.join(save_path, f'sub{sub_id}_session{sess_id}_data.npy')
            lp = os.path.join(save_path, f'sub{sub_id}_session{sess_id}_label.npy')
            np.save(dp, segments)
            np.save(lp, labels)

            # Stats
            cls_dist = {EMOTION_MAP[i]: int((labels == i).sum()) for i in range(4)}
            elapsed = time.time() - t0
            print(f'{segments.shape[0]} windows | {cls_dist} | {elapsed:.1f}s')

            stats[(sub_id, sess_id)] = segments.shape[0]
            total_windows += segments.shape[0]

    elapsed_total = time.time() - t_global
    print(f'\n  Total: {total_windows} windows | {elapsed_total:.0f}s')
    return stats


def verify_output(save_path: str) -> bool:
    """Verify all expected .npy files and print summary."""
    print('\n' + '=' * 60)
    print('VERIFICATION')
    print('=' * 60)

    total = 0
    missing = []
    per_sub = {}

    for sub in range(1, NUM_SUBJECTS + 1):
        sub_total = 0
        for sess in range(1, NUM_SESSIONS + 1):
            dp = os.path.join(save_path, f'sub{sub}_session{sess}_data.npy')
            lp = os.path.join(save_path, f'sub{sub}_session{sess}_label.npy')

            if not os.path.exists(dp):
                missing.append(f'sub{sub}_session{sess}')
                continue

            d = np.load(dp)
            l = np.load(lp)

            assert d.shape[1:] == (NUM_CHANNELS, WINDOW_SIZE), \
                f'sub{sub}_sess{sess}: bad shape {d.shape}'
            assert len(d) == len(l)
            assert set(l).issubset({0, 1, 2, 3})
            assert d.dtype == np.float32
            sub_total += len(d)

        per_sub[sub] = sub_total
        total += sub_total

    # Print per-subject summary
    print(f'  {"Sub":>4} {"Windows":>8} {"Per-Sess":>10}')
    print(f'  {"---":>4} {"-------":>8} {"--------":>10}')
    for sub in range(1, NUM_SUBJECTS + 1):
        n = per_sub.get(sub, 0)
        print(f'  {sub:4d} {n:8d} {n/3:10.1f}')

    print(f'\n  Total windows : {total}')
    print(f'  Missing files : {len(missing)}')
    if missing:
        for m in missing:
            print(f'    - {m}')

    ok = len(missing) == 0
    print(f'  Status        : {"PASS" if ok else "PARTIAL"}')
    return ok


def main():
    parser = argparse.ArgumentParser(
        description='SEED-IV Optimized Preprocessing Pipeline'
    )
    parser.add_argument('--raw-path', type=str,
                        default='data/seed4_raw',
                        help='Root path to raw .mat files')
    parser.add_argument('--save-path', type=str,
                        default='data/seed4_preprocessed',
                        help='Output path for .npy files')
    parser.add_argument('--inspect', action='store_true',
                        help='Only inspect .mat files')
    parser.add_argument('--no-filter', action='store_true',
                        help='Skip bandpass/notch filtering')
    parser.add_argument('--no-artifact-reject', action='store_true',
                        help='Skip artifact rejection')
    args = parser.parse_args()

    # Also check V1's data location
    alt_path = r'C:\Users\dpala_c8xsp2b\Desktop\EEG-TransNet_SEEDIV\data\seed4_raw'
    if not os.path.exists(args.raw_path) and os.path.exists(alt_path):
        print(f'  [INFO] Using V1 data location: {alt_path}')
        args.raw_path = alt_path

    print('=' * 60)
    print('SEED-IV Optimized Preprocessing Pipeline')
    print('=' * 60)
    print(f'  Raw path    : {args.raw_path}')
    print(f'  Save path   : {args.save_path}')
    print(f'  Filtering   : {"ON (0.5-75Hz BP + 50Hz Notch)" if not args.no_filter else "OFF"}')
    print(f'  Artifact rej: {"ON (threshold={:.0f})".format(ARTIFACT_THRESHOLD) if not args.no_artifact_reject else "OFF"}')
    print(f'  Window      : {WINDOW_SIZE} samples = {WINDOW_SEC}s @ {FS}Hz')
    print(f'  Window hop  : {WINDOW_STRIDE} samples ({WINDOW_STRIDE / FS:.1f}s, 50% overlap)')
    print('  Normalization: per-window per-channel z-score (dataset loader)')

    if not os.path.exists(args.raw_path):
        print(f'\n  [ERROR] Path not found: {args.raw_path}')
        sys.exit(1)

    stats = process_all(
        args.raw_path, args.save_path,
        use_filter=not args.no_filter,
        use_artifact_rejection=not args.no_artifact_reject,
        inspect_only=args.inspect,
    )

    if not args.inspect and stats:
        verify_output(args.save_path)


if __name__ == '__main__':
    main()
