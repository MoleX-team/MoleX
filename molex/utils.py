"""
Utility functions for MoleX.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc


def calculate_accuracy(y_true, y_pred):
    """
    Calculate classification accuracy.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        
    Returns:
        float: Accuracy score
    """
    return np.sum(y_true == y_pred) / len(y_true)


def calculate_roc_auc(y_true, y_proba):
    """
    Calculate ROC AUC score.
    
    Args:
        y_true (array-like): True binary labels
        y_proba (array-like): Predicted probabilities
        
    Returns:
        float: ROC AUC score
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return auc(fpr, tpr)


def handle_nan_labels(labels):
    """
    Handle NaN values in labels by replacing with 0.
    
    Args:
        labels (array-like): Labels that may contain NaN values
        
    Returns:
        numpy.ndarray: Labels with NaN values replaced by 0
    """
    labels = np.array(labels)
    return np.nan_to_num(labels.astype(float)).astype(int)


def calculate_statistics(values):
    """
    Calculate mean and standard deviation of values.
    
    Args:
        values (array-like): List of values
        
    Returns:
        tuple: (mean, std)
    """
    values = np.array(values)
    return np.mean(values), np.std(values)