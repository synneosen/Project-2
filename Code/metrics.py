import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)


def confusion_matrix(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm