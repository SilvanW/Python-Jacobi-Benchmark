from pydantic import BaseModel


class Settings(BaseModel):
    width: int
    nx: int
    dt: float
    t_end: int


app_settings = Settings(width=1, nx=100, dt=0.01, t_end=1)
