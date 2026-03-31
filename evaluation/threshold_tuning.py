import numpy as np
from sklearn.metrics import roc_curve


def find_best_threshold(y_true, y_pred):
    """
    Find the best classification threshold per class using Youden's J statistic.
    Youden's J = TPR - FPR  (maximise the gap between sensitivity and false alarm rate)

    If a class has only one unique label value in y_true (e.g. all-negative in a
    small test set), AUC and the ROC curve are undefined — we fall back to 0.5
    for that class instead of returning inf or crashing.

    Args:
        y_true  (np.ndarray): Ground-truth binary labels  shape (N, num_classes)
        y_pred  (np.ndarray): Predicted probabilities      shape (N, num_classes)

    Returns:
        list[float]: Best threshold for each class (length = num_classes)
    """
    best_thresholds = []

    for i in range(y_true.shape[1]):
        # Guard: need both positive and negative samples to compute ROC
        unique_labels = np.unique(y_true[:, i])
        if len(unique_labels) < 2:
            print(f"  Class {i}: only one label value present — using fallback threshold 0.5")
            best_thresholds.append(0.5)
            continue

        fpr, tpr, thresholds = roc_curve(y_true[:, i], y_pred[:, i])

        # Youden's J statistic
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)

        best_thresh = float(thresholds[best_idx])

        # Extra guard: roc_curve can produce threshold values > 1.0 (sentinel values).
        # Clamp to [0.01, 0.99] to keep thresholds meaningful.
        best_thresh = float(np.clip(best_thresh, 0.01, 0.99))

        best_thresholds.append(best_thresh)

    return best_thresholds