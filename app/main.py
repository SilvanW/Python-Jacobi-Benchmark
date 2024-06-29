"""
Entrypoint for the Jacobi Benchmark
"""

import numpy as np
import matplotlib.pyplot as plt
from modules.jacobi import jacobi_component_numpy
from modules.heat_equation import generate_discretization

if __name__ == "__main__":
    dim = 10
    A = generate_discretization(dim)
    x_prev = np.zeros((dim, 1))
    x_curr = np.zeros((dim, 1))
    b = np.zeros((dim, 1))

    b[0] = 0
    b[-1] = 1

    jacobi_component_numpy(A, b, x_prev, x_curr)

    print(x_curr[:, 0])

    plt.plot(x_curr[:, 0], "b-", label=r"$u$")
    plt.ylim(0, 1)
    plt.show()
