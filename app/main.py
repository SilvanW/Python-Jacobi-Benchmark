"""
Entrypoint for the Jacobi Benchmark
"""

import numpy as np
from modules.jacobi import jacobi_component_numpy

if __name__ == "__main__":
    A = np.array([[0.7, -0.1, -0.1], [-0.1, 0.6, -0.1], [-0.1, -0.1, 0.9]])
    b = np.array([[20], [40], [0]])
    x_prev = np.zeros((3, 1))
    x_curr = np.zeros((3, 1))

    jacobi_component_numpy(A, b, x_prev, x_curr)

    print(x_curr)
