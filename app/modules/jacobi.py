"""
Jacobi Solver: Ax = b
"""

import numpy as np
import torch
from numba import jit
from pydantic import BaseModel


class JacobiSettings(BaseModel):
    """
    Settings for the Jacobi Solver
    """

    max_iterations: int = 500
    residual: float = 1e-5


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

    return iteration_count


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

    return iteration_count


def jacobi_matrix_pytorch(
    left_side: torch.Tensor,
    right_side: torch.Tensor,
    previous_solution: torch.Tensor,
    current_solution: torch.Tensor,
    settings: JacobiSettings = JacobiSettings(),
) -> torch.Tensor:
    """Iterative Jacobi Solver using Matrix based approach implmented using pytorch.

    Args:
        left_side (torch.Tensor): Left Side of the Linear System of Equations (A)
        right_side (torch.Tensor): Right Side of the Linear System of Equations (b)
        previous_solution (torch.Tensor): Solution Vector from the Previous Iteration
        current_solution (torch.Tensor): Solution Vector from the Current Iteration
        settings (JacobiSettings, optional): Settings for the Solver. Defaults to JacobiSettings().

    Returns:
        torch.Tensor: Solution of the Linear System of Equations (x)
    """
    iteration_count = 0
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        torch.set_default_device(mps_device)
    else:
        print("GPU Device not found")

    print(f"Device: {torch.get_default_device()}")

    diag = torch.diagflat(torch.diag(left_side))

    diag_inv = torch.diagflat(1 / torch.diag(left_side))

    while (
        torch.norm(torch.matmul(left_side, previous_solution) - right_side)
        > settings.residual
        and iteration_count < settings.max_iterations
    ):

        current_solution = torch.matmul(
            diag_inv, torch.matmul((diag - left_side), previous_solution)
        ) + torch.matmul(diag_inv, right_side)

        previous_solution = current_solution
        iteration_count += 1

    print(iteration_count)

    return current_solution


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
