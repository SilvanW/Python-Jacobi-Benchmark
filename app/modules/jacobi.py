"""
Jacobi: Ax = b
"""

import numpy as np


def jacobi_component(
    a: np.ndarray, b: np.ndarray, x_prev: np.ndarray, x_curr: np.ndarray
) -> None:
    k = 0
    n = 3
    sub_sum = 0
    while np.linalg.norm(np.matmul(a, x_prev) - b) > 10e-5 and k < 100:
        for i in range(0, n):
            for j in range(0, n):
                if j == i:
                    continue
                sub_sum += a[i][j] * x_prev[j][0]

            x_curr[i][0] = (1 / a[i][i]) * (b[i][0] - sub_sum)
            sub_sum = 0

        x_prev = x_curr
        k += 1

    print(k)


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
