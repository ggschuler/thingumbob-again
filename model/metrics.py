import numpy as np

def sensitivity_specificity(true_labels, predicted_labels):
    """
    Compute sensitivity and specificity given true labels and predicted labels.

    Parameters:
    true_labels (list): Array of true labels (0 or 1).
    predicted_labels (list): Array of predicted labels (0 or 1)
    
    Returns:
    sensitivity (float): Sensitivity score.
    specificity (float): Specificity score.
    """
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    TP = np.sum((true_labels == 1) & (predicted_labels == 1))
    FN = np.sum((true_labels == 1) & (predicted_labels == 0))
    TN = np.sum((true_labels == 0) & (predicted_labels == 0))
    FP = np.sum((true_labels == 0) & (predicted_labels == 1))
    sensitivity = TP / (TP + FN) if (TP+FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN+FP) > 0 else 0
    
    return sensitivity, specificity

def recall_precision(true_labels, predicted_labels):
    """
    Compute recall and precision given true labels and predicted labels.

    Parameters:
    true_labels (list): Array of true labels (0 or 1).
    predicted_labels (list): Array of predicted labels (0 or 1).

    Returns:
    recall (float): Recall score.
    precision (float): Precision score.
    """
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    TP = np.sum((true_labels == 1) & (predicted_labels == 1))
    FN = np.sum((true_labels == 1) & (predicted_labels == 0))
    FP = np.sum((true_labels == 0) & (predicted_labels == 1))
    recall = TP / (TP + FN) if (TP) > 0 else 0
    precision = TP / (TP + FP) if (TP) > 0 else 0

    return recall, precision