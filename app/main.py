"""
Entrypoint for the Jacobi Benchmark
"""

import torch
import numpy as np
from modules.jacobi import (
    jacobi_component_numpy,
    jacobi_component_numba,
    jacobi_matrix_pytorch,
)

if __name__ == "__main__":
    A = np.array([[0.7, -0.1, -0.1], [-0.1, 0.6, -0.1], [-0.1, -0.1, 0.9]])
    b = np.array([[20], [40], [0]])
    x_prev = np.zeros((3, 1))
    x_curr = np.zeros((3, 1))

    jacobi_component_numpy(A, b, x_prev, x_curr)

    print(x_curr)

    print("-" * 75)

    A = np.array([[0.7, -0.1, -0.1], [-0.1, 0.6, -0.1], [-0.1, -0.1, 0.9]])
    b = np.array([[20], [40], [0]])
    x_prev = np.zeros((3, 1))
    x_curr = np.zeros((3, 1))

    jacobi_component_numba(A, b, x_prev, x_curr, 500, 10e-5)

    print(x_curr)

    print("-" * 75)

    A = torch.tensor([[0.7, -0.1, -0.1], [-0.1, 0.6, -0.1], [-0.1, -0.1, 0.9]])
    b = torch.tensor([[20.0], [40.0], [0.0]])
    x_prev = torch.zeros((3, 1))
    x_curr = torch.zeros((3, 1))

    solution = jacobi_matrix_pytorch(A, b, x_prev, x_curr)

    print(solution)
