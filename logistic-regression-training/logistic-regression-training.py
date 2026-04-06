import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0
    for i in range(steps):
        p = pred(X, w, b)
        w, b = grad_descend(X, p, y, w, b, lr)
    return w, b

def pred(X, w, b, sig=_sigmoid):
    z = X@w + b
    return sig(z)

def grad_descend(X, p, y, w, b, lr):
    N,_ = X.shape
    error = p-y
    delta_w = (1/N) * X.T@error
    delta_b = np.mean(error)

    w_ = w - lr * delta_w
    b_ = b - lr * delta_b
    return w_, b_