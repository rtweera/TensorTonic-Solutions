import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    x_t_ = np.array(x_t)
    h_prev_ = np.array(h_prev)
    Wx_ = np.array(Wx)
    Wh_ = np.array(Wh)
    b_ = np.array(b)

    return np.tanh(x_t_@Wx_ + h_prev_@Wh_ + b_)
