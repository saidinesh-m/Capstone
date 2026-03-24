import torch
import numpy as np
from tqdm import tqdm
from evaluation.metrics import compute_metrics
import os

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, save_dir='checkpoints'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_auc = 0.0

    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            # Save predictions for metrics (optional for training, but good for tracking)
            all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
            all_targets.append(labels.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(loader.dataset)
        return epoch_loss

    def validate(self, loader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(loader, desc="Validation")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                
                all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                all_targets.append(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(loader.dataset)
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        from evaluation.threshold_tuning import find_best_threshold

        best_thresholds = find_best_threshold(all_targets, all_preds)

        print("\nBest Thresholds per Class:")
        for i, t in enumerate(best_thresholds):
            print(f"Class {i}: {t:.3f}")

        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        plt.figure()

        for i in range(all_targets.shape[1]):
            fpr, tpr, _ = roc_curve(all_targets[:, i], all_preds[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

        # Plot diagonal line (random model)
        plt.plot([0, 1], [0, 1], 'k--')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Multi-Class)')
        plt.legend()

        plt.savefig('roc_curve.png')
        print("ROC curve saved as roc_curve.png")

        plt.close()
                
        metrics = compute_metrics(all_targets, all_preds)
        return epoch_loss, metrics

    def train(self, train_loader, valid_loader, num_epochs, early_stopping_patience=5):
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            train_loss = self.train_epoch(train_loader)
            valid_loss, metrics = self.validate(valid_loader)
            
            if self.scheduler:
                self.scheduler.step(valid_loss)
            
            print(f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | AUC: {metrics['auc']:.4f}")
            
            # Checkpointing
            if metrics['auc'] > self.best_auc:
                self.best_auc = metrics['auc']
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
                print("Best model saved!")
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
                
        print("Training complete.")
