import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # x_ for vector x
    
    w_ = np.array(w)
    g_ = np.array(g)
    s_ = np.array(s)

    s_t = beta*s_ + (1-beta)*g_**2
    w_t = w_ - lr * g_ / (np.sqrt(s_t + eps))

    return (w_t, s_t)