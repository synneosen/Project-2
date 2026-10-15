"""
FFNN

Conventions (ikke endre på noe her uten å informere resten, altså etter alle funksjoer er definert osv.):
- Samples are stored in the rows of X: X has shape (n_samples, n_features).
- Layer l has weight matrix W of shape (n_inputs, n_outputs) and bias b of shape (n_outputs,).
- The feed-forward step of a layer is  z = a @ W + b,  a_next = activation(z).
- Parameters are a list of (W, b) tuples, one per layer.
- Gradients have exactly the same structure: a list of (dW, db) tuples.
"""

import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_der(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def relu(z):
    return np.maximum(0.0, z)


def relu_der(z):
    return np.where(z > 0, 1.0, 0.0)


def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)


def leaky_relu_der(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)

def linear(z):
    """Linear output layer for regression."""
    return z


def linear_der(z):
    return np.ones_like(z)


def softmax(z):
    """Row-wise softmax. Subtract the row max before exp for numerical stability."""
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


# Name -> (function, derivative). Softmax has no entry for its derivative:
# combined with cross entropy, the output error is simply a - y.
ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_der),
    "relu": (relu, relu_der),
    "leaky_relu": (leaky_relu, leaky_relu_der),
    "identity": (linear, linear_der),
    "softmax": (softmax, None),
}

#Cost functions. y_pred and y_true have shape (n_samples, n_outputs).
#For classification, y_true is one-hot encoded.

def mse(y_pred, y_true):
    n = y_true.shape[0]
    return np.sum((y_pred - y_true) ** 2) / n


def mse_der(y_pred, y_true):
    """Derivative of the MSE with respect to y_pred."""
    n = y_true.shape[0]
    return 2.0 * (y_pred - y_true) / n


def cross_entropy(y_pred, y_true, eps=1e-12):
    """Multiclass cross entropy, y_pred = softmax output."""
    n = y_true.shape[0]
    return -np.sum(y_true * np.log(np.clip(y_pred, eps, 1.0))) / n


COSTS = {
    "mse": (mse, mse_der),
    "cross_entropy": (cross_entropy, None),
}


# FFNN network:

#TODO: William sitt starting point 
class NeuralNetwork:
    def __init__(self, layer_sizes, activations, cost="mse", l1=0.0, l2=0.0, seed=2026):
        """
        layer_sizes: e.g. [1, 50, 50, 1] -> input, two hidden layers, output.
                     [784, 10] -> zero hidden layers (logistic regression).
        activations: list of names from ACTIVATIONS, one per layer after the input,
                     so len(activations) == len(layer_sizes) - 1.
                     e.g. ["sigmoid", "sigmoid", "identity"].
        cost:        name from COSTS.
        l1, l2:      penalty strengths lambda. Only affect the gradients in backward().
        seed:        seed for weight initialisation.

        Should create self.params: list of (W, b) tuples.
        Weights: normal distribution, width scaled by the number of inputs
        (Xavier for sigmoid, He for ReLU). Biases: decide and document the choice.
        """
        raise NotImplementedError

    def forward(self, X):
        """
        X: (n_samples, n_features).
        Returns the output, shape (n_samples, n_outputs).
        Should store z and a for every layer, since backward() needs them.
        """
        raise NotImplementedError

    def backward(self, X, y):
        """
        X: (n_samples, n_features), y: (n_samples, n_outputs).
        Runs forward(X), then backpropagation.
        Returns a list of (dW, db) tuples, same order and shapes as self.params.
        Includes the L1/L2 terms if l1 or l2 > 0.
        """
        raise NotImplementedError

    def predict(self, X):
        """Regression: same as forward. Classification: class index per sample."""
        raise NotImplementedError

    def cost(self, X, y):
        """Value of the cost function on (X, y), without the penalty terms."""
        raise NotImplementedError


def train(model, optimizer, X, y, epochs, batch_size=None, seed=2026, X_val=None, y_val=None):
    """
    Trains the model in place.
    batch_size=None -> plain (full-batch) gradient descent.
    batch_size=k    -> stochastic gradient descent with minibatches of size k,
                       reshuffled every epoch with the given seed.
    Returns a dict with the cost per epoch, e.g. {"train": [...], "val": [...]}.
    """
    raise NotImplementedError