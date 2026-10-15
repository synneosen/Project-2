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
        
        self.learningrate = learning_rate #Storing learning rate
    
    def step(self, params, grads):
        
        new_params = [] #Empty list
        for (W,b), (dW, db) in zip(params, grads): #Weights, biases and gradients for one neural network
            W_new = W - self.learningrate*dW #Calculating the new weight
            b_new = b - self.learningrate*db #Calculating the new bias
            new_params.append((W_new, b_new)) #Storing them in the empty list

            return new_params 
        


class RMSprop: #crating a signal for this optimizer
    def __init__(self, learning_rate=1e-3, rho=0.9, eps=1e-8):
        self.learning_rate = learning_rate #Storing the learning rate 
        self.rho = rho #Storing how much of the old memory we keep
        self.eps = eps #Storing a small number to avoid zero divinatin
        self.s = None #Memmory wich is not stored yet

    def step(self, params, grads): 
        if self.s in None: #check if this is the first layer
            #list with memory for every layer
            self.s = [np.zeros_like(W), np.zeros_like(b) for W,b in params] #The 0 array have the shape of W and b, for every layer
        new_params = [] #Epty list for new parameters
        new_s = [] #Empty list for updated memory
        for (W,b), (dW, db), (sW, sb) in zip(params, grads, self.s): #Fetching weights, biases their gradients and memory for every layer
            sW = self.rho*sW + (1-self.rho)*dW**2 #Adding squared gradient to the memory
            sb = self.rho*sb + (1-self.rho)*db**2 #Adding squared gradient to the memory

            W_new = W - self.learning_rate*dW / (np.sqrt(sW) + self.eps) #New weight
            b_new = b - self.learning_rate*db / (np.sqrt(sb) + self.eps) #New bias

            new_params.append((W_new, b_new)) #Add to the parameter list
            new_s.append((sW, sb)) #Add to the memory list
        self.s = new_s #Keep the memory until nex step() run
        return new_s


class Adam:
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate #Store learningrate
        self.beta1 = beta1 #Store how much of the old gradient memory we keep
        self.beta2 = beta2 #Store how much of the square of the old gradient we keeep
        self.eps = eps #Store a small number to avoid zero div

        self.m = None #Memory for gradient, not made yet
        self.v = None #Memory for sq grads, not made yet
        self.t = 0 #Count updates

    def step(self, params, grads):
        """Remember the bias correction with the iteration counter t."""
        if self.m is None:
            self.m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
            self.t += 1
            new_params = []
            new_m = []
            new_v = []

            for (W, b), (dW, db), (mW,mb), (vW,vb) in zip(params, grads, self.m, self.v):

                mW = self.beta1*mW + (1-self.beta1) * dW
                mb = self.beta1*mb + (1-self.beta1) * db

                vW = self.beta2*vW + (1-self.beta2) * dW**2
                vb = self.beta2*vb + (1-self.beta2) * db**2
                #Correction since memories start at zero
                mW_hat = mW/(1 - self.beta1**self.t)
                mb_hat = mb/(1 - self.beta1**self.t)
                vW_hat = vW/(1 - self.beta2**self.t)
                vb_hat = vb/(1 - self.beta2**self.t)
                #Calculating new weigts and biases
                W_new = W - self.learning_rate*mW_hat / (np.sqrt(vW_hat) + self.eps)
                b_new = b - self.learning_rate*mb_hat / (np.sqrt(vb_hat) + self.eps)
                #collecting the new terms
                new_params.append((W_new, b_new))
                new_m.append((mW, mb))
                new_v.append((vW, vb))
            self.m = new_m
            self.v = new_v
            return new_params






        