import numpy as np

def accuracy(y_true, y_pred):
    """Return the fraction of correct predictions using sklearn argument order.

    Inputs are integer class labels, not one-hot encoded targets.
    """
    return np.mean(y_pred == y_true)


def confusion_matrix(y_true, y_pred, n_classes=10):
    """Return a fixed n_classes square matrix for classes 0..n_classes-1.

    Rows are true classes and columns are predicted classes.
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm