"""
This Module provides functionality required to solve the heat equation
"""

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


if __name__ == "__main__":
    print(generate_discretization(3))
