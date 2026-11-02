"""
FFNN
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
    return z


def linear_der(z):
    return np.ones_like(z)


def softmax(z):
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
    n = y_true.shape[0]
    return 2.0 * (y_pred - y_true) / n


def cross_entropy(y_pred, y_true, eps=1e-12):
    n = y_true.shape[0]
    return -np.sum(y_true * np.log(np.clip(y_pred, eps, 1.0))) / n


COSTS = {
    "mse": (mse, mse_der),
    "cross_entropy": (cross_entropy, None),
}


# FFNN network:
class NeuralNetwork:
    def __init__(self, layer_sizes, activations, cost="mse", l1=0.0, l2=0.0, seed=2026):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.cost_name = cost
        self.l1 = l1
        self.l2 = l2

        np.random.seed(seed)
        self.params = [] #the list in which the initial W, b parameters will be stored as tuples for each layer

        for i in range(len(layer_sizes) - 1):
            #shape of W and b determined by input and output size of the layer
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]

            #He for ReLU, Xavier for other activations
            if activations[i] == "relu":
                scale = np.sqrt(2 / n_in)
            else:
                scale = np.sqrt(1 / n_in)

            W = np.random.randn(n_in, n_out) * scale #the weights are picked randomly from scaled normal distribution
            b = np.zeros((1, n_out))  #bias starts at 0 and gets updated through training. 0 used because simple, reliable choice. 

            self.params.append((W, b))

    def forward(self, X):
        self.a = [X] #first a stored as the initial input 
        self.z = []

        a = X

        #loop through the layers and calculate z and then a. Append these to z and a to keep history for backward()
        for i in range(len(self.params)):
            W, b = self.params[i]

            z = a @ W + b
            a = ACTIVATIONS[self.activations[i]][0](z)

            self.z.append(z)
            self.a.append(a)

        return a #makes it simpler to retrieve the final output

    def backward(self, X, y):
        output = self.forward(X)
        n = X.shape[0] #number of input data points

        #derivative of the cost function dC/da
        delta = COSTS[self.cost_name][1](output, y)

        #dC/da times derivative of the last activation function da/dz = derivative of the cost function w.r.t. z at the last activation
        delta = delta * ACTIVATIONS[self.activations[-1]][1](self.z[-1])

        gradients = [] #list for storing gradient at each activation 

        #going backwards through the layers
        for i in reversed(range(len(self.params))):
            W, b = self.params[i]

            dW = self.a[i].T @ delta #dC/dW = dC/dz * dz/dW = delta * a(i-1) (which here has index a[i]). Since initial input is stored as a[0], layer[0] gets 
                                         #a[0] as input, meaning input a from previous layer (i-1) here gets indexed a[i].
            db = np.sum(delta, axis=0, keepdims=True) #dC/db = dC/dz * dz/db = delta

            #regularization
            dW += self.l1 * np.sign(W) #C_total = C + l1*sum(|W|) => (dC/dW)_total = dC/dW + l1*sign(W). Can push weights to zero.  
            dW += self.l2 * W #C_total = C + (l2/2)*sum(W^2) => (dC/dW)_total = dC/dW + l2*W. Penalizes large weights

            gradients.insert(0, (dW, db))

            #in next layer gradient is given by d
            if i > 0:
                delta = delta @ W.T #in next layer (i-1) we get dC/da(i-1) = delta * dz(i)/da(i-1) = delta * W(i)
                delta = delta * ACTIVATIONS[self.activations[i-1]][1](self.z[i-1]) #dC/dz(i-1) = delta * W(i) * da(i-1)/dz(i-1). 
                                                                                   #In next layer this is multiplied by dz(i-1)/dW(i-1) = a(i-2).

        return gradients

    def predict(self, X):
        """Regression: same as forward. Classification: class index per sample."""
        output = self.forward(X)

        if self.cost_name == "mse": #when we use MSE we need raw predictions
            return output 

        if output.shape[1] == 1: #if column vector, return 1D vector with 1 for values >= 0.5 and 0 for <0.5
            return (output >= 0.5).astype(int).flatten()

        return np.argmax(output, axis=1) #if we have more columns, return each row index with the highest score

    def cost(self, X, y):
        """Value of the cost function on (X, y), without the penalty terms."""
        output = self.forward(X)
        return COSTS[self.cost_name][0](output, y) 


def train(model, optimizer, X, y, epochs, batch_size=None, seed=2026, X_val=None, y_val=None):
    """
    Trains the model in place.
    batch_size=None -> plain (full-batch) gradient descent.
    batch_size=k    -> stochastic gradient descent with minibatches of size k,
                       reshuffled every epoch with the given seed.
    Returns a dict with the cost per epoch, e.g. {"train": [...], "val": [...]}.
    """
    history = {"train": [], "val": []} #store cost per epoch
    n = X.shape[0] 
    rng = np.random.default_rng(seed)

    if batch_size is None:
        batch_size = n

    for epoch in range(epochs):

        #shuffle sample indices at the start of each epoch
        indices = rng.permutation(n)

        #shuffle X and y in the same order
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        #split data into batches
        for start in range(0, n, batch_size):
            end = start + batch_size

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            #calculate gradients using backpropagation
            gradients = model.backward(X_batch, y_batch)

            #optimizer returns updated weights and biases
            model.params = optimizer.step(model.params, gradients)

        #calculate training cost after the epoch
        train_cost = model.cost(X, y)
        history["train"].append(train_cost)

        #calculate validation cost if validation data is provided
        if X_val is not None and y_val is not None:
            val_cost = model.cost(X_val, y_val)
            history["val"].append(val_cost)

    return history