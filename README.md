# Project-2

## Group members
- Synne Hjertager Osenbroch
- William Finsnes Naper
- Tora Elise Jaros Madsen

## Description
A feed-forward neural network is implemented from scratch with backpropagation.
It is applied to regression on the one-dimensional Runge function and compared
with OLS, Ridge and Lasso from project 1. It is then applied to multiclass
classification on the full MNIST data set and compared with logistic regression.

## Installation

## Structure

 `Code/nn.py` - Neural network, activation functions, cost functions and training loop 
 `Code/optimizers.py`  Gradient descent, RMSprop and Adam 
 `Code/data.py`  Shared data: Runge data, scaling and MNIST loading 