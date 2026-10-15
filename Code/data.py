import numpy as np
from sklearn.model_selection import train_test_split

seed = 2000

#Legg inn verdiene som gav den beste MSE i prosjekt 1:
# n = 
# sigma = 
test_size = 0.2
seed = 2026

def runge(x):
    return 1.0 / (1.0 + 25*x**2)

#Denne skal lage data på samme måte osm i prosjekt 1 samme med scale tror jeg::
def make_runge_data(n, sigma, test_size, seed):
    "Dette er noisy runge data som er delt inn i train og test"
    "Returnerer x_train, x_test, y_train, y_test"
    "Skalering gjort med scale_regression()"
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    y = runge(x) + rng.normal(0.0, sigma, n)
    X_train, X_test, y_train, y_test = train_test_split(
        x.reshape(-1, 1), y.reshape(-1, 1), test_size=test_size, random_state=seed
    )
    return X_train, X_test, y_train, y_test

def scale_regression(X_train, X_test):
    """Standardise with the mean and std of the training data."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    return (X_train - mean) / std, (X_test - mean) / std
 
def load_mnist(path="data/mnist.npz", val_size = 10000, seed=2026):
    """
    Henter hele MNIST-datasettet.
    Hvert bilde (28x28 piksler) gjøres om til én rad med 784 tall,
    og pikselverdiene deles på 255 så de ligger mellom 0 og 1.
    En del av treningsdataene settes av som valideringssett.
    Valideringssettet brukes til å teste ulike innstillinger (læringsrate, lag, noder osv.).
    Testsettet brukes bare én gang, helt til slutt, for å rapportere resultatet.
    Returnerer tre par: trening, validering og test.
    Fasiten (labels) er tall fra 0 til 9.
    """

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
    """
    Gjør om fasiten fra tall til vektorer.
    Hvert tall blir en rad med 10 plasser, der det står 1 på riktig plass og 0 ellers.
    """    
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out