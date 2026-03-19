import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

def compute_metrics(y_true, y_pred, threshold=0.3): #threshold=0.5):
    """
    Compute metrics for multi-label classification.
    Args:
        y_true (np.array): Ground truth labels (N, NumClasses).
        y_pred (np.array): Predicted probabilities (N, NumClasses).
        threshold (float): Threshold for binary classification.
    Returns:
        dict: Dictionary containing mean AUC, F1, Accuracy, etc.
    """
    metrics = {}
    
    # Example for multi-label: Compute AUC for each class and take mean
    try:
        if y_true.shape[1] > 1:
            # Macro average AUC
            metrics['auc'] = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')
        else:
            # Binary case
            metrics['auc'] = roc_auc_score(y_true, y_pred)
    except ValueError:
        metrics['auc'] = 0.0 # Handle cases with only one class present in batch

    # Binarize predictions
    y_pred_bin = (y_pred > threshold).astype(int)
    
    metrics['f1'] = f1_score(y_true, y_pred_bin, average='macro', zero_division=0)
    metrics['accuracy'] = accuracy_score(y_true, y_pred_bin)
    metrics['precision'] = precision_score(y_true, y_pred_bin, average='macro', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred_bin, average='macro', zero_division=0)
    
    return metrics
