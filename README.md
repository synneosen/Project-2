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
From the repository root, install the required packages with:
python -m pip install -r requirements.txt

Download MNIST from
https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
and place the file at `data/mnist.npz`.

## Run the baseline
From the repository root, run:

    python Code/logreg_baseline.py
    

This trains logistic regression on the training split, evaluates on the validation split (not the test set), prints the result, and writes `results/logreg_baseline.json`. Training on all 50,000 images may take a while.

## Tests
Run from the `Code` folder:

    python -m unittest discover

## Structure

 `Code/nn.py` - Neural network, activation functions, cost functions and training loop 
 `Code/optimizers.py`  Gradient descent, RMSprop and Adam 
 `Code/data.py`  Shared data: Runge data, scaling and MNIST loading 
 `Code/plotting.py` Plotts the results
 `Code/metrics.py` Accuracy and confusion matrix
 `Code/logreg_baseline.py` Trains and evaluates the sklearn MNIST logistic-regression baseline
 `results/logreg_baseline.json` Stores baseline metrics and training time after running the script

## Notebooks
The notebooks in the code folder explore regression and classificationwith the neural network implemented in this project. 

-part_b_regression.ipynb

-part_cd_activations_regularization.ipynb
Investigates regression on the Runge function. It compares sigmoid, ReLU and
Leaky ReLU across different network depths and widths, then evaluates L1 and
L2 regularization strengths and learning rates. The notebook displays
training- and validation-MSE heatmaps, regularization heatmaps, and prints
the selected configurations and test-set results.

- `part_e_classification_an.ipynb`  
Investigates multiclass classification on MNIST. It compares Adam and RMSprop,
tests learning rates and network architectures, and examines L2
regularization. It reports cross-entropy and accuracy, displays loss plots
and a confusion matrix, and compares the neural network with scikit-learn
logistic regression.
