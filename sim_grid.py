# pylint: disable=import-error
from config import Config
import numpy as np
from space import Space

class StdRes(Space):
    """
    Homogenous resolution space
    """

    def setup(self, cfg: Config):

        arraylen = cfg.steps[0] * cfg.steps[1]

        neighbors = np.array(
            [
                np.zeros((arraylen,), dtype=int),  # Up
                np.zeros((arraylen,), dtype=int),  # Down
                np.zeros((arraylen,), dtype=int),  # Left
                np.zeros((arraylen,), dtype=int),  # Right
            ]
        )

        dx = np.ones(
            (
                2,
                arraylen,
            )
        )
        dx[0] = cfg.dx[0]
        dx[1] = cfg.dx[1]
        
        values = np.zeros((4, arraylen))

        for y in range(cfg.steps[1]):
            for x in range(cfg.steps[0]):

                neighbors[:, self.coordinate_to_index(cfg, x, y)] = [
                    self.coordinate_to_index(cfg, x, y + 1),
                    self.coordinate_to_index(cfg, x, y - 1),
                    self.coordinate_to_index(cfg, x + 1, y),
                    self.coordinate_to_index(cfg, x - 1, y),
                ]

        boundary = []

        for x in range(cfg.steps[0]):
            for y in [0,cfg.steps[1] - 1]:
                boundary += [(x, y)]

        for y in range(cfg.steps[1]):
            for x in [0,cfg.steps[0] - 1]:
                boundary += [(x, y)]

        boundary = np.array([self.coordinate_to_index(cfg, *v) for v in boundary])
        store = {"arraylen": arraylen}

        return neighbors, boundary, dx, store

    def external_setup(self, cfg):
        return np.zeros((self.store["arraylen"],))

    def coordinate_to_index(self, cfg, x, y):
        return (x % cfg.steps[0]) + cfg.steps[0] * (y % cfg.steps[1])

    def from_grid(self, cfg):
        SqMatrix = np.array(
            [
                cfg.list_p,  # density
                cfg.list_p * cfg.list_v[0],  # momentum x
                cfg.list_p * cfg.list_v[1],  # momentum y
                (
                    cfg.list_e
                    + 0.5 * cfg.list_p * np.sum(cfg.list_v, 0) ** 2
                ),  # energy
            ]
        )

        values = np.zeros((4, self.store["arraylen"]))

        for y in range(cfg.steps[1]):
            for x in range(cfg.steps[0]):
                values[:, self.coordinate_to_index(cfg, x, y)] = SqMatrix[:, x, y]
        
        return values

    def to_grid(self, U):
        tmp = np.zeros((4, *self.cfg.steps))
        for y in range(self.cfg.steps[1]):
            for x in range(self.cfg.steps[0]):
                tmp[:, x, y] = U[:, self.coordinate_to_index(self.cfg, x, y)]

        return tmp

    def data_in_dir(self, data, dir, dist=1):
        n = self.neighbors[dir]
        for _ in range(dist-1):
            n = self.neighbors[dir][n]
        return data.T[n].T