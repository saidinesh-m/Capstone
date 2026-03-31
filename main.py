import argparse
import os
import torch
import pandas as pd
import cv2
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix as sk_cm

from models.densenet_cxr import DenseNet121
from data.dataset import CXRDataset
from preprocessing.transforms import get_train_transforms, get_valid_transforms
from training.trainer import Trainer
from utils.loss import FocalLoss
from explainability.gradcam import GradCAMExplainer
from evaluation.threshold_tuning import find_best_threshold

CLASSES = [
    'slvh',
    'dlv',
    'composite_slvh_dlv',
    'heart_transplant',
    'lung_transplant',
    'pacemaker_or_icd'
]

SHORT_NAMES = ['slvh', 'dlv', 'composite', 'heart_tx', 'lung_tx', 'pacemaker']
OUTPUT_DIR  = r"O:\Capstone\ConfusionMatrix"


def get_id_col(df):
    if 'cxr_filename' in df.columns:
        return 'cxr_filename'
    elif 'cal_filename' in df.columns:
        return 'cal_filename'
    elif 'Image Index' in df.columns:
        return 'Image Index'
    return df.columns[0]


def filter_valid_images(df, image_dir, id_col):
    valid_indices = []
    missing_count = 0
    for idx, row in df.iterrows():
        img_id = os.path.basename(str(row[id_col]).strip())
        possible_paths = [
            os.path.join(image_dir, img_id),
            os.path.join(image_dir, img_id + '.png'),
            os.path.join(image_dir, img_id + '.jpg'),
            os.path.join(image_dir, img_id + '.jpeg'),
        ]
        if any(os.path.exists(p) for p in possible_paths):
            valid_indices.append(idx)
        else:
            missing_count += 1
    return valid_indices, missing_count


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    BATCH_SIZE    = args.batch_size
    LEARNING_RATE = args.lr
    NUM_EPOCHS    = args.epochs
    NUM_CLASSES   = len(CLASSES)

    # ------------------------------------------------------------------
    # MODEL SETUP (shared across all modes)
    # ------------------------------------------------------------------
    model = DenseNet121(num_classes=NUM_CLASSES, is_trained=True).to(device)

    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights from {args.weights}")

    # ------------------------------------------------------------------
    # TRAIN MODE
    # ------------------------------------------------------------------
    if args.mode == 'train':
        if not os.path.exists(args.csv_path):
            print(f"Error: CSV not found at {args.csv_path}")
            return

        df = pd.read_csv(args.csv_path)
        df.columns = df.columns.str.strip()
        id_col = get_id_col(df)

        print("Checking for image availability...")
        valid_indices, missing_count = filter_valid_images(df, args.image_dir, id_col)
        print(f"Found {len(valid_indices)} images. Missing {missing_count}.")
        df = df.loc[valid_indices]

        if len(df) == 0:
            print("Error: No images found.")
            return

        train_df = df.sample(frac=0.7, random_state=42)
        valid_df = df.drop(train_df.index)
        print(f"Train size: {len(train_df)}, Valid size: {len(valid_df)}")

        train_dataset = CXRDataset(train_df, args.image_dir, transforms=get_train_transforms(), classes=CLASSES)
        valid_dataset = CXRDataset(valid_df, args.image_dir, transforms=get_valid_transforms(), classes=CLASSES)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        criterion = FocalLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)

        trainer = Trainer(model, criterion, optimizer, scheduler, device)
        trainer.train(train_loader, valid_loader, NUM_EPOCHS)

    # ------------------------------------------------------------------
    # PREDICT MODE
    # ------------------------------------------------------------------
    elif args.mode == 'predict':
        if not args.image_path:
            print("Error: --image_path required for predict mode.")
            return

        model.eval()
        image = cv2.imread(args.image_path)
        if image is None:
            print("Could not read image.")
            return

        image_rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transforms = get_valid_transforms()
        augmented  = transforms(image=image_rgb)
        img_tensor = augmented['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs   = torch.sigmoid(outputs).cpu().numpy()[0]

        print("\nPredictions:")
        for lab, prob in zip(CLASSES, probs):
            print(f"  {lab:25s}: {prob:.4f}")

        target_layer = model.densenet121.features[-1]
        explainer    = GradCAMExplainer(model, target_layer)
        rgb_norm     = cv2.resize(image_rgb, (224, 224)) / 255.0
        heatmap      = explainer.explain(img_tensor, rgb_norm, target_class=None)

        out_path = "gradcam_output.png"
        cv2.imwrite(out_path, (heatmap * 255).astype(np.uint8))
        print(f"Grad-CAM heatmap saved to {out_path}")

    # ------------------------------------------------------------------
    # TEST MODE  (100% of found images, no splitting)
    # ------------------------------------------------------------------
    elif args.mode == 'test':
        if not os.path.exists(args.csv_path):
            print(f"Error: CSV not found at {args.csv_path}")
            return

        if not args.weights:
            print("Error: --weights required for test mode.")
            return

        df = pd.read_csv(args.csv_path)
        df.columns = df.columns.str.strip()
        id_col = get_id_col(df)

        valid_indices, missing_count = filter_valid_images(df, args.image_dir, id_col)
        print(f"Found {len(valid_indices)} valid images. Missing: {missing_count}.")
        df = df.loc[valid_indices]

        if len(df) == 0:
            print("ERROR: No valid images found.")
            return

        print(f"Testing on ALL {len(df)} images.")

        test_dataset = CXRDataset(df, args.image_dir, transforms=get_valid_transforms(), classes=CLASSES)
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        criterion = FocalLoss()
        trainer   = Trainer(model, criterion, None, None, device)
        _, metrics = trainer.validate(test_loader)

        print("\nTest Results (ALL IMAGES):")
        for k, v in metrics.items():
            if not np.isnan(v):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: N/A")

    # ------------------------------------------------------------------
    # EVALUATE MODE  (70/30 split — mirrors training split)
    # ------------------------------------------------------------------
    elif args.mode == 'evaluate':
        if not os.path.exists(args.csv_path):
            print(f"Error: CSV not found at {args.csv_path}")
            return

        if not args.weights:
            print("Error: --weights required for evaluate mode.")
            return

        df = pd.read_csv(args.csv_path)
        df.columns = df.columns.str.strip()
        id_col = get_id_col(df)

        valid_indices, missing_count = filter_valid_images(df, args.image_dir, id_col)
        print(f"Found {len(valid_indices)} valid images. Missing: {missing_count}.")
        df = df.loc[valid_indices]

        if len(df) == 0:
            print("Error: No valid images found after filtering.")
            return

        train_df = df.sample(frac=0.7, random_state=42)
        valid_df = df.drop(train_df.index)
        print(f"Evaluating on {len(valid_df)} images (30% validation set).")

        valid_dataset = CXRDataset(valid_df, args.image_dir, transforms=get_valid_transforms(), classes=CLASSES)
        valid_loader  = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        criterion = FocalLoss()
        optimizer = optim.AdamW(model.parameters())
        trainer   = Trainer(model, criterion, optimizer, None, device)
        _, metrics = trainer.validate(valid_loader)

        print("\nEvaluation Results:")
        for k, v in metrics.items():
            if not np.isnan(v):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: N/A")

    # ------------------------------------------------------------------
    # TEST PER CLASS MODE  (16 images per class, 6 groups = 96 total)
    # ------------------------------------------------------------------
    elif args.mode == 'test_per_class':
        if not os.path.exists(args.csv_path):
            print(f"Error: CSV not found at {args.csv_path}")
            return

        if not args.weights:
            print("Error: --weights required.")
            return

        df = pd.read_csv(args.csv_path)
        df.columns = df.columns.str.strip()
        id_col = get_id_col(df)

        valid_indices, missing_count = filter_valid_images(df, args.image_dir, id_col)
        print(f"Found {len(valid_indices)} valid images. Missing: {missing_count}.")
        df = df.loc[valid_indices].reset_index(drop=True)

        if len(df) == 0:
            print("ERROR: No valid images found.")
            return

        # Split into 6 groups of 16 — one group per class
        images_per_class = 16
        class_dfs = {}
        for i, cls in enumerate(CLASSES):
            start = i * images_per_class
            end   = start + images_per_class
            class_dfs[cls] = df.iloc[start:end].copy()
            print(f"  {cls:25s}: rows {start}–{end-1}  ({len(class_dfs[cls])} images)")

        model.eval()
        all_preds_total   = []
        all_targets_total = []

        print("\n── Per-class test results (16 images each) ─────────────────")

        for i, (cls, cls_df) in enumerate(class_dfs.items()):
            print(f"\n  Class: {cls}  (rows {i*16}–{i*16+15})")

            dataset = CXRDataset(cls_df, args.image_dir,
                                 transforms=get_valid_transforms(), classes=CLASSES)
            loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=4)

            preds_list   = []
            targets_list = []

            with torch.no_grad():
                for images_batch, labels in loader:
                    images_batch = images_batch.to(device)
                    outputs      = model(images_batch)
                    probs        = torch.sigmoid(outputs).cpu().numpy()
                    preds_list.append(probs)
                    targets_list.append(labels.numpy())

            preds   = np.vstack(preds_list)
            targets = np.vstack(targets_list)

            all_preds_total.append(preds)
            all_targets_total.append(targets)

            # AUC — target class only
            target_col = targets[:, i]
            pred_col   = preds[:, i]

            if len(np.unique(target_col)) < 2:
                print(f"    AUC ({cls}): N/A — no positive samples in this 16-image group")
            else:
                auc_val = roc_auc_score(target_col, pred_col)
                print(f"    AUC ({cls}): {auc_val:.4f}")

            # 2x2 confusion matrix — target class only
            thresh = find_best_threshold(targets, preds)[i]
            binary = (preds[:, i] >= thresh).astype(int)

            if len(np.unique(target_col)) >= 2:
                cm             = sk_cm(target_col, binary, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                recall         = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                precision      = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                f1             = (2 * precision * recall / (precision + recall)
                                  if (precision + recall) > 0 else 0.0)
                print(f"    TP={tp}  TN={tn}  FP={fp}  FN={fn}")
                print(f"    Recall={recall:.2f}  Precision={precision:.2f}  F1={f1:.2f}")
            else:
                tn_count = int(np.sum(target_col == 0))
                print(f"    TP=0  TN={tn_count}  FP=N/A  FN=0  (no positives in group)")

        # Build 6x6 matrix across all 96 images combined
        print("\n── 6×6 Confusion Matrix across all 96 images ───────────────")
        all_preds_full   = np.vstack(all_preds_total)
        all_targets_full = np.vstack(all_targets_total)

        best_thresholds  = find_best_threshold(all_targets_full, all_preds_full)
        binary_preds_all = (all_preds_full >= np.array(best_thresholds)).astype(int)

        num_classes = all_targets_full.shape[1]
        cm_6x6      = np.zeros((num_classes, num_classes), dtype=int)
        for ii in range(num_classes):
            for jj in range(num_classes):
                cm_6x6[ii, jj] = int(np.sum(
                    (all_targets_full[:, ii] == 1) & (binary_preds_all[:, jj] == 1)
                ))

        # Overall AUC — only classes that have both positives and negatives
        valid_aucs = []
        for ii in range(num_classes):
            if len(np.unique(all_targets_full[:, ii])) >= 2:
                valid_aucs.append(
                    roc_auc_score(all_targets_full[:, ii], all_preds_full[:, ii])
                )
        overall_auc = np.mean(valid_aucs) if valid_aucs else 0.0

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
            f'6×6 Confusion Matrix — all 96 images  |  AUC = {overall_auc:.4f}',
            fontsize=13, fontweight='bold', pad=12
        )
        plt.xticks(rotation=30, ha='right', fontsize=10)
        plt.yticks(rotation=0,  fontsize=10)
        plt.tight_layout()

        path_6x6 = os.path.join(OUTPUT_DIR, f"confusion_matrix_6x6_FULL_{timestamp}.png")
        plt.savefig(path_6x6, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"6×6 matrix saved:         {path_6x6}")
        print(f"Overall AUC (96 images):  {overall_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CXR Cardiovascular Detection System")
    parser.add_argument('--mode',       type=str,   required=True,
                        choices=['train', 'predict', 'evaluate', 'test', 'test_per_class'],
                        help='Mode: train | predict | evaluate | test | test_per_class')
    parser.add_argument('--csv_path',   type=str,   default='data/train_labels.csv')
    parser.add_argument('--image_dir',  type=str,   default='data/images')
    parser.add_argument('--image_path', type=str,   help='Path to single image (predict mode only)')
    parser.add_argument('--weights',    type=str,   help='Path to trained model weights (.pth)')
    parser.add_argument('--epochs',     type=int,   default=25)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-4)

    args = parser.parse_args()
    main(args)