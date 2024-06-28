"""
Jacobi: Ax = b
"""

import numpy as np
from pydantic import BaseModel


class JacobiSettings(BaseModel):
    max_iterations: int = 500
    residual: float = 10e-5


def jacobi_component(
    a: np.ndarray,
    b: np.ndarray,
    x_prev: np.ndarray,
    x_curr: np.ndarray,
    settings: JacobiSettings = JacobiSettings(),
) -> None:
    iteration_count = 0
    dimension = x_prev.shape[0]
    sub_sum = 0
    while (
        np.linalg.norm(np.matmul(a, x_prev) - b) > settings.residual
        and iteration_count < settings.max_iterations
    ):
        for i in range(0, dimension):
            for j in range(0, dimension):
                if j == i:
                    continue
                sub_sum += a[i][j] * x_prev[j][0]

            x_curr[i][0] = (1 / a[i][i]) * (b[i][0] - sub_sum)
            sub_sum = 0

        x_prev = x_curr
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

    jacobi_component(A, b, x_prev, x_curr)

    print(np.linalg.norm(np.matmul(A, x_curr) - b))

    print(x_curr)
