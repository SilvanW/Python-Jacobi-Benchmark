"""
This Module provides functionality required to solve the heat equation
"""

from typing import Tuple
import numpy as np


def generate_discretization(dimension: int) -> np.ndarray:
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


def generate_equation_system(
    dimension: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A = generate_discretization(dimension)
    x_prev = np.zeros((dimension, 1))
    x_curr = np.zeros((dimension, 1))
    b = np.zeros((dimension, 1))

    b[0] = 0
    b[-1] = 1

    return A, x_prev, x_curr, b


if __name__ == "__main__":
    print(generate_discretization(3))
