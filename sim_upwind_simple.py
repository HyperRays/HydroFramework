# pylint: disable=import-error
# # Relaxing Upwind implementation in Python
# 
# Authors: Soham Kuvalekar, Daniel Iancu
# Date: 18/07/2024
# 
# This program provides a simple implementation of a relaxing Upwind
# hydrodynamics simulation using numpy, to be used in an external 
# framework
# 
# Based on paper:
# Trac, H., & Pen, U. L. (2003). A primer on eulerian
# computational fluid dynamics for astrophysics. Publications of \
# Astronomical Society of Pacific, 115(805), 303.
# DOI: https://doi.org/10.1086/367747
# 
# Our special thanks to our Maturarbeit supervisor: M. Liebendörfer
# 


import numpy as np
from space import Space
import logging

class Simulation:
    def __init__(self, space: Space):

        self.uses_torch = False
        assert space.uses_torch == self.uses_torch

        self.space = space
        self.U = self.space.U_vector()
        self.all_states = [self.U]
        self.all_times = [0]
        self.cfl_violated = False

    ## Current implementation of Upwind: Relaxing Upwind
    def calc_F(self, U, axis):

        if axis == 0:
            leftidx = 2  # Left
            rightidx = 3  # Right
        else:
            leftidx = 0  # Up
            rightidx = 1  # Down

        v = U[1] / U[0]  # velocity at cell center in sweep direction

        epsilon = U[3] - 0.5 * (np.sum(U[1:3] ** 2, 0) / U[0])
        P = (self.space.cfg.gamma - 1) * epsilon
        P = np.maximum(P, 0)

        c = np.abs(v) + np.maximum(
            np.sqrt(self.space.cfg.gamma * P / U[0]), 1e-12
        )

        w = U * v
        w[1] += P
        w[3] += P * v

        fr = (U * c + w) / 2

        ## To differenciate what is in other cells
        # Temporary values for neigboring cells are labeled with curr_

        curr_fl = (U * c - w) / 2

        # Take value from right cell
        fl = self.space.data_in_dir(curr_fl[:], rightidx)

        dF = fr - fl

        # Take value from left cell
        dF_neighbor = self.space.data_in_dir(dF[:], leftidx)

        dFdx = (dF - dF_neighbor) / self.space.dx[axis]

        # suppress effect of periodic boundary conditions
        if not self.space.cfg.spherical:
            # dFdx[:, [0, -1]] = 0
            dFdx[:, self.space.boundary] = 0

        return dFdx

    def sweepx(self, U, sweepc):
        dFdx = self.calc_F(U, 0)
        dU = self.space.external - dFdx
        return U + (dU * self.space.cfg.dt * 1 / sweepc)

    def sweepy(self, U, sweepc):
        dFdx = self.calc_F(U[[0, 2, 1, 3]], 1)[[0, 2, 1, 3]]
        dU = self.space.external - dFdx
        return U + (dU * self.space.cfg.dt * 1 / sweepc)

    def calc_next_U(self, U):

        Unext = U.copy()

        # Sweep count
        sweepc = 2

        Unext = self.sweepx(Unext, sweepc)
        Unext = self.sweepy(Unext, sweepc)
        Unext = self.sweepy(Unext, sweepc)
        Unext = self.sweepx(Unext, sweepc)
        
        if np.any(np.isnan(Unext)) or np.any(np.isinf(Unext)):
            print("Invalid values detected in Unext")

        return Unext
        


    def compute_cfl(self, U):
        velocities = np.abs(U[1:3] / U[0]).sum(0)  # Fluid velocity

        # Calculate pressure using epsilon (internal energy)
        epsilon = U[3] - 0.5 * (U[0] * velocities**2)
        P = (self.space.cfg.gamma - 1) * epsilon
        P = np.maximum(P, 0)

        # Speed of sound calculation
        sound_speed = np.maximum(
            np.sqrt(self.space.cfg.gamma * P / U[0]), 1e-5
        )

        cfl = (velocities + sound_speed) * self.space.cfg.dt / max(self.space.cfg.dx)

        # calculate valid timestep
        valid_dt = min(self.space.cfg.dx) / (velocities + sound_speed)

        return cfl.max(), valid_dt.min()

    def run(self, target_time, start_time=0, checkpoint_freq=1, verbose=True, no_iter=False, auto_count=50):

        logger = logging.getLogger(__name__)

        if checkpoint_freq == "auto":
            _,dt = self.compute_cfl(self.U)
            iterc = target_time/dt
            checkpoint_freq = max(int(iterc/auto_count),1)

            logger.info(f"Set checkpoint frequency to {checkpoint_freq}")

        # Adjust start time if any set

        assert start_time < target_time
        self.all_times[0] += start_time
        
        highest_egy = np.sum(
            self.U[3]
        )  # if all energy was concentrated in one cell (very improbable)
        low_mass = max(
            np.min(self.U[0]), 1e-5
        )  # and that mass was min of start config (with so much energy very improbable)
        highest_v = np.sqrt(
            2 * highest_egy / low_mass
        )  # and all of that energy was kinetic energy
        smallest_dt = (
            np.min(self.space.cfg.dx) / np.maximum(highest_v, 1e-20)
        )  # this would be a very pessimistic dt
        self.space.cfg.dt = smallest_dt
        t = int(target_time / smallest_dt + 0.5)
        t_steps = 0
        logger.info(f"Max number of iterations: {t}")

        total_time = start_time

        if no_iter:
            t = 0

        for i in range(t):

            cfl_number, valid_dt = self.compute_cfl(self.U)
            self.space.cfg.dt = min(valid_dt * 0.3, target_time - total_time)

            if np.isnan(valid_dt):
                break

            total_time = total_time + self.space.cfg.dt

            t_steps += 1

            self.U = self.calc_next_U(self.U)
            
            info_txt = f"iter: {i}, dt: {self.space.cfg.dt}, total: {total_time}"
            
            if i%checkpoint_freq == 0:
                self.all_states.append(self.U)
                self.all_times.append(total_time)
                info_txt += f" [Saved]"
                
            if verbose:
                logger.info(info_txt)

            if total_time >= target_time:
                logger.info("Completed")
                break

            if cfl_number * 0.7 > 1 and not self.cfl_violated:
                logger.warning(
                    f"CFL condition violated at step {i}, CFL number = {cfl_number}"
                )
                self.cfl_violated = True

        if self.all_times[-1] != total_time:
            self.all_states.append(self.U)
            self.all_times.append(total_time)

        return np.array(self.all_states), np.array(self.all_times)
