"""
Entrypoint for the Jacobi Benchmark
"""

import matplotlib.pyplot as plt
import timeit
import numpy as np
from modules.heat_equation import generate_equation_system
from modules.jacobi import jacobi_component_numpy, jacobi_component_numba

if __name__ == "__main__":
    dim = 50

    A, x_prev, x_curr, b = generate_equation_system(dim)

    numpy_start = timeit.default_timer()
    iteration_count_numpy = jacobi_component_numpy(A, b, x_prev, x_curr)
    execution_time_numpy = timeit.default_timer() - numpy_start

    print(
        f"Numpy Solver finished in {execution_time_numpy:05.2f} seconds with {iteration_count_numpy} iterations"
    )

    A, x_prev, x_curr, b = generate_equation_system(dim)

    numba_start = timeit.default_timer()
    iteration_count_numba = jacobi_component_numba(A, b, x_prev, x_curr, 500, 1e-5)
    execution_time_numba = timeit.default_timer() - numba_start

    print(
        f"Numba Solver finished in {execution_time_numba:05.2f} seconds with {iteration_count_numba} iterations"
    )

    """
    plt.plot(x_curr[:, 0], "b-", label=r"$u$")
    plt.ylim(0, 1)
    plt.show()
    """
