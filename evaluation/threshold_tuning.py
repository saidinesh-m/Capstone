import numpy as np
from sklearn.metrics import roc_curve

def find_best_threshold(y_true, y_pred):
    best_thresholds = []

    for i in range(y_true.shape[1]):  # loop over classes
        fpr, tpr, thresholds = roc_curve(y_true[:, i], y_pred[:, i])
        
        # Youden’s J statistic
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        
        best_thresh = thresholds[best_idx]
        best_thresholds.append(best_thresh)

    return best_thresholds