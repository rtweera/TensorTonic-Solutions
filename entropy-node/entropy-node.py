import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    labels, counts = np.unique(y, return_counts=True)
    total_f = np.sum(counts)
    p = counts/total_f
    return -np.sum(p * np.log2(p))
    