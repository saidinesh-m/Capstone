import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

CLASSES = ['slvh', 'dlv', 'composite_slvh_dlv',
           'heart_transplant', 'lung_transplant', 'pacemaker_or_icd']

def plot_confusion_matrices(y_true, y_pred_bin, save_path='confusion_matrix.png'):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    colors = {
        'TP': '#1D9E75', 'TN': '#639922',
        'FP': '#D85A30', 'FN': '#BA7517'
    }

    for i, cls in enumerate(CLASSES):
        ax = axes[i]
        cm = confusion_matrix(y_true[:, i], y_pred_bin[:, i], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        vals  = [[tp, fn],  [fp, tn]]
        tags  = [['TP', 'FN'], ['FP', 'TN']]
        clrs  = [[colors['TP'], colors['FN']], [colors['FP'], colors['TN']]]

        for r in range(2):
            for c in range(2):
                ax.add_patch(plt.Rectangle((c, 1-r), 1, 1,
                             color=clrs[r][c], alpha=0.85))
                ax.text(c + 0.5, 1 - r + 0.55, str(vals[r][c]),
                        ha='center', va='center',
                        fontsize=22, fontweight='bold', color='white')
                ax.text(c + 0.5, 1 - r + 0.25, tags[r][c],
                        ha='center', va='center',
                        fontsize=11, color='white', alpha=0.9)

        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(['Pred Positive', 'Pred Negative'], fontsize=10)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(['Actual Negative', 'Actual Positive'], fontsize=10)
        ax.set_title(f'{cls}\nRecall={recall:.2f}  Precision={precision:.2f}  F1={f1:.2f}',
                     fontsize=11, pad=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.suptitle('Confusion Matrix — per class', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix image saved to {save_path}")


def compute_metrics(y_true, y_pred, threshold=0.3983):
    metrics = {}

    try:
        if y_true.shape[1] > 1:
            metrics['auc'] = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')
        else:
            metrics['auc'] = roc_auc_score(y_true, y_pred)
    except ValueError:
        metrics['auc'] = 0.0

    y_pred_bin = (y_pred > threshold).astype(int)

    metrics['f1']        = f1_score(y_true, y_pred_bin, average='macro', zero_division=0)
    metrics['accuracy']  = accuracy_score(y_true, y_pred_bin)
    metrics['precision'] = precision_score(y_true, y_pred_bin, average='macro', zero_division=0)
    metrics['recall']    = recall_score(y_true, y_pred_bin, average='macro', zero_division=0)

    # ── Per-class AUC ─────────────────────────────────────────────────
    print("\n── Per-Class AUC ──────────────────────────────────────")
    for i, cls in enumerate(CLASSES):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            auc = 0.0
        print(f"  {cls:25s}  AUC = {auc:.4f}")

    # ── Confusion Matrix printed + saved as PNG ───────────────────────
    print("\n── Confusion Matrix (per class) ───────────────────────")
    print(f"  {'Class':25s}  {'TP':>5}  {'TN':>5}  {'FP':>5}  {'FN':>5}")
    print(f"  {'-'*25}  {'-----':>5}  {'-----':>5}  {'-----':>5}  {'-----':>5}")
    for i, cls in enumerate(CLASSES):
        try:
            cm = confusion_matrix(y_true[:, i], y_pred_bin[:, i], labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
        except ValueError:
            tn = fp = fn = tp = 0
        print(f"  {cls:25s}  {tp:>5}  {tn:>5}  {fp:>5}  {fn:>5}")

    plot_confusion_matrices(y_true, y_pred_bin, save_path='confusion_matrix.png')

    return metrics