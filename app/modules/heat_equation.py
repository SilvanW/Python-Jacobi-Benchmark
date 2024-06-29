"""
This Module provides functionality required to solve the heat equation
"""

import torch
from typing import Tuple
import numpy as np


def generate_discretization_np(dimension: int) -> np.ndarray:
    """Generates 1D Difference Star, which is the Left side of the Heat equation

    Args:
        dimension (int): Dimension of the Left Side

    Returns:
        np.ndarray: Left Side Matrix widht shape (dimension, dimension)
    """
    discretisation_matrix = np.zeros((dimension, dimension))

    np.fill_diagonal(discretisation_matrix, 2)

    for i in range(dimension - 1):
        discretisation_matrix[i, i + 1] = -1

    for i in range(1, dimension):
        discretisation_matrix[i, i - 1] = -1

    return discretisation_matrix


def generate_dicretization_torch(dimension: int) -> torch.Tensor:
    discretisation_tensor = torch.zeros((dimension, dimension))

    discretisation_tensor = discretisation_tensor.fill_diagonal_(2)

    for i in range(dimension - 1):
        discretisation_tensor[i, i + 1] = -1

    for i in range(1, dimension):
        discretisation_tensor[i, i - 1] = -1

    return discretisation_tensor


def generate_equation_system_np(
    dimension: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A = generate_discretization_np(dimension)
    x_prev = np.zeros((dimension, 1))
    x_curr = np.zeros((dimension, 1))
    b = np.zeros((dimension, 1))

    b[0] = 0.0
    b[-1] = 1.0

    return A, x_prev, x_curr, b


def generate_equation_system_torch(
    dimension: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    A = generate_dicretization_torch(dimension)
    x_prev = torch.zeros((dimension, 1))
    x_curr = torch.zeros((dimension, 1))
    b = torch.zeros((dimension, 1))

    b[0] = 0.0
    b[-1] = 1.0

    return A, x_prev, x_curr, b


if __name__ == "__main__":
    print(generate_discretization_np(3))
    print(generate_equation_system_np(2))

    print("-" * 75)

    print(generate_dicretization_torch(3))
    print(generate_equation_system_torch(3))
