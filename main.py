import argparse
import os
import torch
import pandas as pd
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.densenet_cxr import DenseNet121
from data.dataset import CXRDataset
from preprocessing.transforms import get_train_transforms, get_valid_transforms
from training.trainer import Trainer
from utils.loss import FocalLoss
from explainability.gradcam import GradCAMExplainer

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    NUM_EPOCHS = args.epochs
    NUM_EPOCHS = args.epochs
    
    # Updated Class List based on User's CSV
    # Columns seen: slvh, dlv, composite, heart_transplant, lung_transplant, pacemaker_or_icd
    # Note: Adjust these strings if the exact CSV column names differ slightly
    CLASSES = [
        'slvh', 
        'dlv', 
        'composite_slvh_dlv', 
        'heart_transplant', 
        'lung_transplant', 
        'pacemaker_or_icd'
    ]
    NUM_CLASSES = len(CLASSES)
    
    # 1. Dataset Loading
    # Check if csv exists, otherwise assuming dummy or user needs to provide
    if args.mode == 'train':
        if not os.path.exists(args.csv_path):
            print(f"Error: Dataset CSV not found at {args.csv_path}")
            return
            
        df = pd.read_csv(args.csv_path)
        
        # Check if we need to clean headers (remove whitespace)
        df.columns = df.columns.str.strip()
        
        # --- Pre-filter validation ---
        print("Checking for image availability...")
        valid_indices = []
        missing_count = 0
        
        # Identify ID column (logic copied from dataset.py for consistency)
        if 'cxr_filename' in df.columns:
            id_col = 'cxr_filename'
        elif 'cal_filename' in df.columns:
             id_col = 'cal_filename'
        elif 'Image Index' in df.columns:
            id_col = 'Image Index'
        else:
            id_col = df.columns[0]
            
        for idx, row in df.iterrows():
            img_id = str(row[id_col])
            # Check existence (simplified logic matching dataset.py)
            possible_paths = [
                os.path.join(args.image_dir, img_id),
                os.path.join(args.image_dir, img_id + '.png'),
                os.path.join(args.image_dir, img_id + '.jpg'),
                os.path.join(args.image_dir, img_id + '.jpeg'),
                os.path.join(args.image_dir, os.path.basename(img_id)),
                os.path.join(args.image_dir, os.path.basename(img_id) + '.png'),
                os.path.join(args.image_dir, os.path.basename(img_id) + '.jpg'),
                 os.path.join(args.image_dir, os.path.basename(img_id) + '.jpeg')
            ]
            
            if any(os.path.exists(p) for p in possible_paths):
                valid_indices.append(idx)
            else:
                missing_count += 1
                
        print(f"Found {len(valid_indices)} images. Missing {missing_count} images from CSV.")
        df = df.loc[valid_indices]
        
        if len(df) == 0:
            print("Error: No images found from the CSV in the specified directory.")
            return

        # Assuming simple random split for demo purposes
        train_df = df.sample(frac=0.8, random_state=42)
        valid_df = df.drop(train_df.index)
        
        train_dataset = CXRDataset(train_df, args.image_dir, transforms=get_train_transforms(), classes=CLASSES)
        valid_dataset = CXRDataset(valid_df, args.image_dir, transforms=get_valid_transforms(), classes=CLASSES)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        print(f"Train size: {len(train_dataset)}, Valid size: {len(valid_dataset)}")

    # 2. Model Setup
    model = DenseNet121(num_classes=NUM_CLASSES, is_trained=True).to(device)
    
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights from {args.weights}")

    # 3. Training Mode
    if args.mode == 'train':
        criterion = FocalLoss() # or nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)
        
        trainer = Trainer(model, criterion, optimizer, scheduler, device)
        trainer.train(train_loader, valid_loader, NUM_EPOCHS)

    # 4. Inference/Explainability Mode
    elif args.mode == 'predict':
        model.eval()
        image = cv2.imread(args.image_path)
        if image is None:
            print("Could not read image.")
            return
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transforms = get_valid_transforms()
        augmented = transforms(image=image_rgb)
        img_tensor = augmented['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
            
        # Assuming standard NIH labels for demo
        labels = [
            'slvh', 'dlv', 'composite_slvh_dlv', 'heart_transplant', 'lung_transplant', 'pacemaker_or_icd'
        ]
        
        print("\nPredictions:")
        for lab, prob in zip(labels, probs):
            print(f"{lab}: {prob:.4f}")
            
        # Explainer
        # DenseNet feature layer is typically features.denseblock4.denselayer16 (last one) or similar
        # But for torchvision models.densenet121, it is model.densenet121.features[-1]
        target_layer = model.densenet121.features[-1]
        explainer = GradCAMExplainer(model, target_layer)
        
        # Normalize original image to [0,1] for overlay
        rgb_norm = cv2.resize(image_rgb, (224, 224)) / 255.0
        
        heatmap = explainer.explain(img_tensor, rgb_norm, target_class=None)
        
        out_path = "gradcam_output.png"
        cv2.imwrite(out_path, (heatmap * 255).astype(np.uint8))
        print(f"Explainability heatmap saved to {out_path}")


    # 5. Evaluation Mode
    elif args.mode == 'evaluate':
        # Re-using dataset loading logic partially
        if not os.path.exists(args.csv_path):
            print(f"Error: Dataset CSV not found at {args.csv_path}")
            return
            
        df = pd.read_csv(args.csv_path)
        df.columns = df.columns.str.strip()
        
        # Identify ID column
        if 'cxr_filename' in df.columns:
            id_col = 'cxr_filename'
        elif 'cal_filename' in df.columns:
             id_col = 'cal_filename'
        elif 'Image Index' in df.columns:
            id_col = 'Image Index'
        else:
            id_col = df.columns[0]
            
        # Filter existing images
        valid_indices = []
        for idx, row in df.iterrows():
            img_id = str(row[id_col])
            possible_paths = [
                os.path.join(args.image_dir, img_id),
                os.path.join(args.image_dir, img_id + '.png'),
                os.path.join(args.image_dir, img_id + '.jpg'),
                os.path.join(args.image_dir, img_id + '.jpeg'),
                os.path.join(args.image_dir, os.path.basename(img_id)),
                os.path.join(args.image_dir, os.path.basename(img_id) + '.png'),
                os.path.join(args.image_dir, os.path.basename(img_id) + '.jpg'),
                 os.path.join(args.image_dir, os.path.basename(img_id) + '.jpeg')
            ]
            if any(os.path.exists(p) for p in possible_paths):
                valid_indices.append(idx)
        
        df = df.loc[valid_indices]
        if len(df) == 0:
            print("Error: No images found.")
            return

        # For evaluation, we ideally want a held-out set. 
        # Here we mimic the split to get the same validation set as training.
        train_df = df.sample(frac=0.8, random_state=42)
        valid_df = df.drop(train_df.index)
        
        print(f"Evaluating on {len(valid_df)} images (Validation Set).")
        
        valid_dataset = CXRDataset(valid_df, args.image_dir, transforms=get_valid_transforms(), classes=CLASSES)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        # Load weights
        if not args.weights:
            print("Error: --weights argument is required for evaluation.")
            return
            
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights from {args.weights}")
        
        # Run validation
        criterion = FocalLoss() # Needed for trainer init, though not strictly for metrics
        optimizer = optim.AdamW(model.parameters()) # Dummy
        scheduler = None
        
        trainer = Trainer(model, criterion, optimizer, scheduler, device)
        _, metrics = trainer.validate(valid_loader)
        
        print("\nEvaluation Results:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CXR Detection System")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict', 'evaluate'], help='Mode: train, predict, or evaluate')
    parser.add_argument('--csv_path', type=str, default='data/train_labels.csv', help='Path to dataset CSV')
    parser.add_argument('--image_dir', type=str, default='data/images', help='Path to image directory')
    parser.add_argument('--image_path', type=str, help='Path to single image for prediction')
    parser.add_argument('--weights', type=str, help='Path to trained weights')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)

    args = parser.parse_args()
    main(args)
