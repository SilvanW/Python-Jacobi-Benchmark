"""
Entrypoint for the Jacobi Benchmark
"""

import matplotlib.pyplot as plt
import timeit
import numpy as np
from modules.heat_equation import (
    generate_equation_system_np,
    generate_equation_system_torch,
)
from modules.jacobi import (
    jacobi_component_numpy,
    jacobi_component_numba,
    jacobi_matrix_pytorch,
)

if __name__ == "__main__":
    dim = 50

    # Numpy
    A, x_prev, x_curr_np, b = generate_equation_system_np(dim)

    numpy_start = timeit.default_timer()
    iteration_count_numpy = jacobi_component_numpy(A, b, x_prev, x_curr_np)
    execution_time_numpy = timeit.default_timer() - numpy_start

    print(
        f"Numpy Solver finished in {execution_time_numpy:05.2f} seconds with {iteration_count_numpy} iterations"
    )

    # Numba
    A, x_prev, x_curr_numba, b = generate_equation_system_np(dim)

    numba_start = timeit.default_timer()
    iteration_count_numba = jacobi_component_numba(
        A, b, x_prev, x_curr_numba, 500, 1e-5
    )
    execution_time_numba = timeit.default_timer() - numba_start

    print(
        f"Numba Solver finished in {execution_time_numba:05.2f} seconds with {iteration_count_numba} iterations"
    )

    # Torch
    A, x_prev, x_curr, b = generate_equation_system_torch(dim)

    torch_start = timeit.default_timer()
    iteration_count_torch, solution = jacobi_matrix_pytorch(A, b, x_prev, x_curr)
    execution_time_torch = timeit.default_timer() - torch_start

    print(
        f"PyTorch Solver finished in {execution_time_torch:05.2f} seconds with {iteration_count_torch} iterations"
    )

    fig, ax = plt.subplots(1, 3)

    # Numpy
    ax[0].plot(x_curr_np[:, 0])
    ax[0].set_title("Numpy")
    ax[0].grid()

    # Numba
    ax[1].plot(x_curr_numba[:, 0])
    ax[1].set_title("Numba")
    ax[1].grid()

    # Torch
    ax[2].plot(solution[:, 0])
    ax[2].set_title("Torch")
    ax[2].grid()

    plt.show()
