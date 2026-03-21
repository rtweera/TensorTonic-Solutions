def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x = x0
    for _ in range(steps):
        x = x - lr * derivative_of_quadratic(x, a, b)
    return x


def derivative_of_quadratic(x, a, b):
    return 2*a*x + b