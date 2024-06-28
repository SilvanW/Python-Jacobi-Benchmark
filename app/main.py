import numpy as np
from pydantic import BaseModel

from modules.jacobi import jacobi_component


class Settings(BaseModel):
    width: int
    nx: int
    dt: float
    t_end: int


app_settings = Settings(width=1, nx=100, dt=0.01, t_end=1)

if __name__ == "__main__":
    A = np.array([[0.7, -0.1, -0.1], [-0.1, 0.6, -0.1], [-0.1, -0.1, 0.9]])
    b = np.array([[20], [40], [0]])
    x_prev = np.zeros((3, 1))
    x_curr = np.zeros((3, 1))

    jacobi_component(A, b, x_prev, x_curr)

    print(x_curr)
