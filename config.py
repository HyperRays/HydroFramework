import numpy as np


class Config:
    """
    Stores config values for any Hydrodynamics Simulation 
    Setup
    """

    def __init__(
        self,
        vol: np.ndarray,
        steps: np.ndarray,
        list_p: np.ndarray,
        list_v: np.ndarray,
        list_e: np.ndarray,
        dt: float,
        gamma: float,
        spherical: bool,
    ):
        self.vol = vol
        self.steps = steps
        self.dx = tuple(v / s for v, s in zip(vol, steps))
        self.list_p = list_p
        self.list_v = list_v
        self.list_e = list_e
        self.gamma = gamma
        self.external = np.zeros((4, *steps))
        self.dt = dt
        self.spherical = spherical