import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.array(A)
    (m, n) = np.shape(A)
    Z = np.zeros((n, m))

    for mi in range(m):
        for ni in range(n):
            print(type(mi), type(ni))
            Z[ni, mi] = A[mi, ni]
    return Z
