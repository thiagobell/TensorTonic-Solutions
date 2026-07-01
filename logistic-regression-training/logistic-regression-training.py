import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    bias = 0
    N, D  = X.shape
    W = np.zeros((D, 1))
    
    y = y.reshape((N, 1))
    
    for s in range(steps):
        # pred has shape (N, D)
        pred = _sigmoid(X @ W + bias)
        # compute the loss
        # derivative of sigmoind = sigmoid * (1-sigmoid)
        # y needs to be in the shape (N, 1)
        delta_w = X.transpose() @ (pred - y) / N
        #delta_w = np.zeros((1, X.shape[1]))
        delta_b = np.mean(pred-y)

        W = W - delta_w * 0.1
        bias = bias - delta_b * 0.1 
    return W.reshape((D,)), float(bias)