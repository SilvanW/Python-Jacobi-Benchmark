"""
Jacobi Solver: Ax = b
"""

import numpy as np
from numba import jit
from pydantic import BaseModel


class JacobiSettings(BaseModel):
    """
    Settings for the Jacobi Solver
    """

    max_iterations: int = 500
    residual: float = 10e-5


def jacobi_component_numpy(
    left_side: np.ndarray,
    right_side: np.ndarray,
    previous_solution: np.ndarray,
    current_solution: np.ndarray,
    settings: JacobiSettings = JacobiSettings(),
) -> None:
    """
    Iterative Jacobi Solver using Component based approach implmented using numpy.
    All Arrays must be passed by reference.

    Args:
        left_side (np.ndarray): Left Side of the Linear System of Equations (A)
        right_side (np.ndarray): Right Side of the Linear System of Equations (b)
        previous_solution (np.ndarray): Solution Vector from the Previous Iteration
        current_solution (np.ndarray): Solution Vector from the Current Iteration
        settings (JacobiSettings, optional): Settings for the Solver. Defaults to JacobiSettings().
    """
    iteration_count = 0
    dimension = previous_solution.shape[0]
    sub_sum = 0
    while (
        np.linalg.norm(np.matmul(left_side, previous_solution) - right_side)
        > settings.residual
        and iteration_count < settings.max_iterations
    ):
        for i in range(0, dimension):
            for j in range(0, dimension):
                if j == i:
                    continue
                sub_sum += left_side[i][j] * previous_solution[j][0]

            current_solution[i][0] = (1 / left_side[i][i]) * (
                right_side[i][0] - sub_sum
            )
            sub_sum = 0

        previous_solution = current_solution
        iteration_count += 1

    print(iteration_count)


@jit()
def jacobi_component_numba(
    left_side: np.ndarray,
    right_side: np.ndarray,
    previous_solution: np.ndarray,
    current_solution: np.ndarray,
    max_iterations: int,
    residual: float,
) -> None:
    """
    Iterative Jacobi Solver using Component based approach implmented using numpy.
    All Arrays must be passed by reference.

    Args:
        left_side (np.ndarray): Left Side of the Linear System of Equations (A)
        right_side (np.ndarray): Right Side of the Linear System of Equations (b)
        previous_solution (np.ndarray): Solution Vector from the Previous Iteration
        current_solution (np.ndarray): Solution Vector from the Current Iteration
        max_iterations (int): Max iteration before stop
        residual (float): Accepted Residual
    """
    iteration_count = 0
    dimension = previous_solution.shape[0]
    sub_sum = 0
    while (
        np.linalg.norm((left_side @ previous_solution) - right_side) > residual
        and iteration_count < max_iterations
    ):
        for i in range(0, dimension):
            for j in range(0, dimension):
                if j == i:
                    continue
                sub_sum += left_side[i][j] * previous_solution[j][0]

            current_solution[i][0] = (1 / left_side[i][i]) * (
                right_side[i][0] - sub_sum
            )
            sub_sum = 0

        previous_solution = current_solution
        iteration_count += 1

    print(iteration_count)


if __name__ == "__main__":
    A = np.array([[0.7, -0.1, -0.1], [-0.1, 0.6, -0.1], [-0.1, -0.1, 0.9]])
    b = np.array([[20], [40], [0]])
    x_prev = np.zeros((3, 1))
    x_curr = np.zeros((3, 1))

    print(A)

    print(x_curr)

    print(b)

    print("-" * 75)

    print(np.linalg.norm(np.matmul(A, x_curr) - b))

    jacobi_component_numpy(A, b, x_prev, x_curr)

    print(np.linalg.norm(np.matmul(A, x_curr) - b))

    print(x_curr)
