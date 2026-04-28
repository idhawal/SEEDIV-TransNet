from __future__ import annotations

"""
Evaluation metrics and visualization for SE-TransNet results.
Confusion matrices, per-subject bar charts, summary reports.
"""

from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

EMOTION_NAMES = ['Neutral', 'Sad', 'Fear', 'Happy']


def print_summary(results: dict, paradigm: str = 'SD') -> None:
    print('\n' + '=' * 65)
    print(f'{paradigm} RESULTS SUMMARY')
    print('=' * 65)
    accs = [r['acc'] for r in results.values()]
    kappas = [r['kappa'] for r in results.values()]
    f1s = [r['f1_macro'] for r in results.values()]
    for sub_id, res in results.items():
        print(f"  Sub {sub_id:02d}: Acc={res['acc']*100:.2f}%  "
              f"F1={res['f1_macro']*100:.2f}%  k={res['kappa']:.4f}")
    print(f"\n  Mean Acc  : {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%")
    print(f"  Mean F1   : {np.mean(f1s)*100:.2f}% +/- {np.std(f1s)*100:.2f}%")
    print(f"  Mean Kappa: {np.mean(kappas):.4f} +/- {np.std(kappas):.4f}")


def plot_confusion_matrix(cm, title='Confusion Matrix', save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTION_NAMES)
    disp1.plot(ax=axes[0], cmap='Blues', colorbar=True)
    axes[0].set_title(f'{title} (counts)')
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=EMOTION_NAMES)
    disp2.plot(ax=axes[1], cmap='Blues', colorbar=True, values_format='.2f')
    axes[1].set_title(f'{title} (normalized)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_subject_accuracies(results, paradigm='SD', save_path=None):
    subjects = sorted(results.keys())
    accs = [results[s]['acc'] * 100 for s in subjects]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(subjects)), accs, color='#5B9BD5',
                  edgecolor='#2F5496', alpha=0.85, width=0.7)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.axhline(mean_acc, color='#C00000', linewidth=2, linestyle='--',
               label=f'Mean: {mean_acc:.2f}%')
    ax.axhspan(mean_acc - std_acc, mean_acc + std_acc, color='#FFB3B3', alpha=0.3,
               label=f'Std: +/-{std_acc:.2f}%')
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels([f'S{s}' for s in subjects], fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Subject ID', fontsize=12)
    ax.set_title(f'SE-TransNet V4: {paradigm} (SEED-IV)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_summary_txt(results, save_path, paradigm='SD', config=None):
    accs = [r['acc'] for r in results.values()]
    kappas = [r['kappa'] for r in results.values()]
    f1s = [r['f1_macro'] for r in results.values()]
    with open(save_path, 'w') as f:
        f.write(f'SE-TransNet V4 -- {paradigm} Results\n')
        f.write('=' * 50 + '\n\n')
        if config:
            for k, v in config.items():
                f.write(f'  {k}: {v}\n')
            f.write('\n')
        for sub_id, res in sorted(results.items()):
            f.write(f"  Sub {sub_id:02d}: acc={res['acc']:.6f}  "
                    f"f1={res['f1_macro']:.6f}  kappa={res['kappa']:.6f}\n")
        f.write(f"\nMean Acc:   {np.mean(accs):.6f} +/- {np.std(accs):.6f}\n")
        f.write(f"Mean F1:    {np.mean(f1s):.6f} +/- {np.std(f1s):.6f}\n")
        f.write(f"Mean Kappa: {np.mean(kappas):.6f} +/- {np.std(kappas):.6f}\n")
