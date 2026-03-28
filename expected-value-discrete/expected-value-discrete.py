import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x_ = np.array(x)
    p_ = np.array(p)

    if np.shape(x_) != np.shape(p_):
        raise ValueError("Dimension mismatch")
    if not np.allclose(np.sum(p_), 1.0):
        raise ValueError("Probabilities don't add up to 1")
    return np.sum(x_ * p_)