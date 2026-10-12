"""
Alle optimizerne brukes på samme måte:
    opt = Adam(learning_rate=1e-3)
    params = opt.step(params, grads)

params er vektene og biasene i nettverket.
grads er gradientene som backward() regner ut.
Begge er lister med (W, b), én for hvert lag, og har samme form.

step() gjør ett oppdateringssteg og gir tilbake de nye vektene.

RMSprop og Adam må huske tidligere gradienter mellom hvert steg.
Dette minnet lages automatisk første gang step() kalles.

Om det blir vanlig gradient descent eller stokastisk gradient descent,
bestemmes av batch_size i train() i nn.py, ikke her.
"""

#TODO: Tora fyll inn 

import numpy as np


class GD:
    """Plain update: theta <- theta - learning_rate * grad."""

    def __init__(self, learning_rate=1e-2):
        raise NotImplementedError

    def step(self, params, grads):
        raise NotImplementedError


class RMSprop:
    def __init__(self, learning_rate=1e-3, rho=0.9, eps=1e-8):
        raise NotImplementedError

    def step(self, params, grads):
        raise NotImplementedError


class Adam:
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        raise NotImplementedError

    def step(self, params, grads):
        """Remember the bias correction with the iteration counter t."""
        raise NotImplementedError