import numpy as np
from scipy.optimize import minimize


def f(x):
    return x[0]**2 + 2 * x[1]**2 + 3 * x[2]**2 + x[0] * x[1] + x[1] * x[2] + 5


def grad_f(x):
    grad = np.zeros_like(x)
    grad[0] = 2 * x[0] + x[1]
    grad[1] = 4 * x[1] + x[0] + x[2]
    grad[2] = 6 * x[2] + x[1]
    return grad


def weighted_f(x, w):
    weighted_x = x * w
    return f(weighted_x)


def weighted_grad_f(x, w):
    weighted_x = x * w
    grad = grad_f(weighted_x)
    return grad * weights


def compute_differences(x):
    kernel = np.array([1, -1])
    diff = np.convolve(x, kernel, mode='valid')
    return diff


x0 = np.array([1.0, 3.0, 10.0])
weights = np.array([1.0, 0.5, 2.0])

result = minimize(weighted_f, x0, args=(weights,), method='BFGS', jac=weighted_grad_f)

print(f"Optimal x: {result.x}, f(x) = {result.fun}")

differences = compute_differences(result.x)
print(f"Differences: {differences}")