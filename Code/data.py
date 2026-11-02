import numpy as np
from sklearn.model_selection import train_test_split

n = 100
sigma = 0.1
test_size = 0.2
seed = 2026

def runge(x):
    return 1.0 / (1.0 + 25*x**2)

def make_runge_data(n, sigma, test_size, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    y = runge(x) + rng.normal(0.0, sigma, n)
    X_train, X_test, y_train, y_test = train_test_split(
        x.reshape(-1, 1), y.reshape(-1, 1), test_size=test_size, random_state=seed
    )
    return X_train, X_test, y_train, y_test

def scale_regression(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    return (X_train - mean) / std, (X_test - mean) / std
 
def load_mnist(path="data/mnist.npz", val_size = 10000, seed=2026):
    with np.load(path) as data:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]
 
    x_train = x_train.reshape(len(x_train), -1) / 255.0
    x_test = x_test.reshape(len(x_test), -1) / 255.0
 
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size, random_state=seed, stratify=y_train
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
 
 
def one_hot(y, n_classes=10):
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out