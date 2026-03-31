import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_curve, auc, confusion_matrix

from evaluation.metrics import compute_metrics

CLASSES     = ['slvh', 'dlv', 'composite_slvh_dlv', 'heart_transplant', 'lung_transplant', 'pacemaker_or_icd']
SHORT_NAMES = ['slvh', 'dlv', 'composite', 'heart_tx', 'lung_tx', 'pacemaker']
OUTPUT_DIR  = r"O:\Capstone\ConfusionMatrix"


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, save_dir='checkpoints'):
        self.model     = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device    = device
        self.save_dir  = save_dir

        os.makedirs(self.save_dir, exist_ok=True)
        self.best_auc = 0.0

    # ------------------------------------------------------------------
    # TRAIN ONE EPOCH
    # ------------------------------------------------------------------
    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss    = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})

        return running_loss / len(loader.dataset)

    # ------------------------------------------------------------------
    # VALIDATE / TEST
    # ------------------------------------------------------------------
    def validate(self, loader):
        self.model.eval()
        running_loss = 0.0
        all_preds    = []
        all_targets  = []

        with torch.no_grad():
            pbar = tqdm(loader, desc="Validation")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss    = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        epoch_loss  = running_loss / len(loader.dataset)
        all_preds   = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # ── Threshold tuning ──────────────────────────────────────────
        from evaluation.threshold_tuning import find_best_threshold
        best_thresholds = find_best_threshold(all_targets, all_preds)

        print("\nBest Thresholds per Class:")
        for i, t in enumerate(best_thresholds):
            print(f"  {CLASSES[i]:25s}: {t:.3f}")

        # ── ROC curve + per-class AUC ─────────────────────────────────
        auc_scores = self._save_roc_curve(all_targets, all_preds, timestamp)

        # ── Confusion matrices (6x6 + per-class 2x2) ─────────────────
        binary_preds = (all_preds >= np.array(best_thresholds)).astype(int)
        self._save_confusion_matrix_6x6(all_targets, binary_preds, auc_scores, timestamp)
        self._save_confusion_matrix_perclass(all_targets, binary_preds, auc_scores, timestamp)

        # ── Compute metrics ───────────────────────────────────────────
        metrics = compute_metrics(all_targets, all_preds)

        return epoch_loss, metrics

    # ------------------------------------------------------------------
    # ROC CURVE
    # ------------------------------------------------------------------
    def _save_roc_curve(self, all_targets, all_preds, timestamp):
        auc_scores = []
        plt.figure(figsize=(8, 6))

        for i in range(all_targets.shape[1]):
            if len(np.unique(all_targets[:, i])) < 2:
                print(f"  {CLASSES[i]}: skipping ROC (only one class present)")
                auc_scores.append(np.nan)
                continue
            fpr, tpr, _ = roc_curve(all_targets[:, i], all_preds[:, i])
            roc_auc     = auc(fpr, tpr)
            auc_scores.append(roc_auc)
            plt.plot(fpr, tpr, label=f'{SHORT_NAMES[i]} (AUC={roc_auc:.2f})')

        valid_aucs   = [a for a in auc_scores if not np.isnan(a)]
        overall_auc  = np.mean(valid_aucs) if valid_aucs else 0.0

        plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
        plt.xlabel('False Positive Rate', fontsize=11)
        plt.ylabel('True Positive Rate',  fontsize=11)
        plt.title(f'ROC Curve  |  Overall AUC = {overall_auc:.4f}', fontsize=12, fontweight='bold')
        plt.legend(fontsize=9)
        plt.tight_layout()

        roc_path = os.path.join(OUTPUT_DIR, f"roc_curve_{timestamp}.png")
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nROC curve saved:   {roc_path}")
        print(f"Overall AUC:       {overall_auc:.4f}")

        return auc_scores

    # ------------------------------------------------------------------
    # 6x6 CONFUSION MATRIX
    # ------------------------------------------------------------------
    def _save_confusion_matrix_6x6(self, all_targets, binary_preds, auc_scores, timestamp):
        num_classes = all_targets.shape[1]
        cm_6x6      = np.zeros((num_classes, num_classes), dtype=int)

        for i in range(num_classes):
            for j in range(num_classes):
                cm_6x6[i, j] = int(np.sum(
                    (all_targets[:, i] == 1) & (binary_preds[:, j] == 1)
                ))

        valid_aucs  = [a for a in auc_scores if not np.isnan(a)]
        overall_auc = np.mean(valid_aucs) if valid_aucs else 0.0

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            cm_6x6, annot=True, fmt='d', cmap='Blues',
            xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
            ax=ax, linewidths=0.5, linecolor='white',
            annot_kws={'size': 12, 'weight': 'bold'}
        )
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('True Class',      fontsize=12)
        ax.set_title(
            f'6×6 Confusion Matrix  |  Overall AUC = {overall_auc:.4f}',
            fontsize=13, fontweight='bold', pad=12
        )
        plt.xticks(rotation=30, ha='right', fontsize=10)
        plt.yticks(rotation=0,  fontsize=10)
        plt.tight_layout()

        path = os.path.join(OUTPUT_DIR, f"confusion_matrix_6x6_{timestamp}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"6×6 confusion matrix saved: {path}")

    # ------------------------------------------------------------------
    # PER-CLASS 2x2 CONFUSION MATRIX (all 6 in one image)
    # ------------------------------------------------------------------
    def _save_confusion_matrix_perclass(self, all_targets, binary_preds, auc_scores, timestamp):
        colors = {
            'TP': '#1D9E75',   # green  — correct positive
            'TN': '#639922',   # olive  — correct negative
            'FP': '#D85A30',   # coral  — false alarm
            'FN': '#BA7517',   # amber  — missed detection
        }

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()

        for i, cls in enumerate(CLASSES):
            ax  = axes[i]
            cm  = confusion_matrix(all_targets[:, i], binary_preds[:, i], labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            vals = [[tp, fn], [fp, tn]]
            tags = [['TP', 'FN'], ['FP', 'TN']]
            clrs = [[colors['TP'], colors['FN']], [colors['FP'], colors['TN']]]

            for r in range(2):
                for c in range(2):
                    ax.add_patch(plt.Rectangle(
                        (c, 1 - r), 1, 1,
                        color=clrs[r][c], alpha=0.88
                    ))
                    ax.text(c + 0.5, 1 - r + 0.60,
                            str(vals[r][c]),
                            ha='center', va='center',
                            fontsize=26, fontweight='bold', color='white')
                    ax.text(c + 0.5, 1 - r + 0.28,
                            tags[r][c],
                            ha='center', va='center',
                            fontsize=12, color='white', alpha=0.92)

            recall    = tp / (tp + fn)  if (tp + fn) > 0  else 0.0
            precision = tp / (tp + fp)  if (tp + fp) > 0  else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)
            auc_val   = auc_scores[i] if not np.isnan(auc_scores[i]) else None
            auc_str   = f"  AUC={auc_val:.3f}" if auc_val is not None else "  AUC=N/A"

            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)
            ax.set_xticks([0.5, 1.5])
            ax.set_xticklabels(['Pred Positive', 'Pred Negative'], fontsize=9)
            ax.set_yticks([0.5, 1.5])
            ax.set_yticklabels(['Actual Negative', 'Actual Positive'], fontsize=9)
            ax.set_title(
                f'{cls}\n'
                f'Recall={recall:.2f}   Precision={precision:.2f}   F1={f1:.2f}{auc_str}',
                fontsize=10, pad=7
            )
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors['TP'], label='TP — correct positive'),
            Patch(facecolor=colors['FN'], label='FN — missed detection'),
            Patch(facecolor=colors['FP'], label='FP — false alarm'),
            Patch(facecolor=colors['TN'], label='TN — correct negative'),
        ]
        fig.legend(
            handles=legend_elements,
            loc='lower center', ncol=4,
            fontsize=10, frameon=False,
            bbox_to_anchor=(0.5, -0.02)
        )

        plt.suptitle(
            'Per-class confusion matrix (2×2)',
            fontsize=14, fontweight='bold', y=1.01
        )
        plt.tight_layout()

        path = os.path.join(OUTPUT_DIR, f"confusion_matrix_perclass_{timestamp}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Per-class confusion matrix saved: {path}")

    # ------------------------------------------------------------------
    # FULL TRAINING LOOP
    # ------------------------------------------------------------------
    def train(self, train_loader, valid_loader, num_epochs, early_stopping_patience=5):
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            train_loss             = self.train_epoch(train_loader)
            valid_loss, metrics    = self.validate(valid_loader)

            if self.scheduler:
                self.scheduler.step(valid_loss)

            auc_val = metrics['auc'] if not np.isnan(metrics['auc']) else 0.0
            print(f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | AUC: {auc_val:.4f}")

            if auc_val > self.best_auc:
                self.best_auc = auc_val
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_dir, 'best_model.pth')
                )
                print("Best model saved!")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        print("Training complete.")