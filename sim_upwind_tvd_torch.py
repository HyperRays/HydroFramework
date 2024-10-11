# pylint: disable=all
# # Relaxing Upwind implementation in Python
#
# Authors: Soham Kuvalekar, Daniel Iancu
# Date: 18/07/2024
#
# This program provides a TVD implementation of a relaxing Upwind
# hydrodynamics simulation using torch, to be used in an external
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
import torch
from space import Space
import logging

def vanleer(flux, a, b):
    c = a * b
    c_ge_0 = c > 0
    flux = flux.clone()
    flux[c_ge_0] = flux[c_ge_0] + 2 * c[c_ge_0] / (a[c_ge_0] + b[c_ge_0])
    return flux

class Simulation:
    def __init__(self, space: Space):
        self.space = space

        self.U = torch.from_numpy(self.space.U_vector())
        self.all_states = [self.U]
        self.all_times = [0]
        self.cfl_violated = False

        # Convert external to torch tensor
        self.external_torch = torch.from_numpy(self.space.external)

        # Ensure boundary indices are in list format for PyTorch indexing
        if isinstance(self.space.boundary, np.ndarray):
            self.space.boundary = self.space.boundary.tolist()

    ## Current implementation of Upwind: Relaxing Upwind
    def calc_F(self, Uinp, axis, sweepc):

        if axis == 0:
            leftidx = 2  # Left
            rightidx = 3  # Right
        else:
            leftidx = 0  # Up
            rightidx = 1  # Down

        def _wrap(U):
            v = U[1] / U[0]  # velocity at cell center in sweep direction

            epsilon = U[3] - 0.5 * (torch.sum(U[1:3] ** 2, dim=0) / U[0])
            P = (self.space.cfg.gamma - 1) * epsilon
            P = torch.maximum(P, torch.zeros_like(P))

            c = torch.abs(v) + torch.maximum(
                torch.sqrt(self.space.cfg.gamma * P / U[0]), torch.tensor(1e-12, dtype=U.dtype, device=U.device)
            )

            w = U * v
            w = w.clone()
            w[1] = w[1] + P
            w[3] = w[3] + P * v

            fr = (U * c + w) / 2

            curr_fl = (U * c - w) / 2

            # Take value from right cell
            fl = self.space.data_in_dir(curr_fl.clone(), rightidx)

            dF = fr - fl

            # Take value from left cell
            dF_neighbor = self.space.data_in_dir(dF.clone(), leftidx)

            dFdx = (dF - dF_neighbor) / self.space.dx[axis]

            # suppress effect of periodic boundary conditions
            if not self.space.cfg.spherical:
                dFdx[:, self.space.boundary] = 0

            return dFdx, c, w

        dFdx, c, w = _wrap(Uinp)
        dU = self.external_torch - dFdx
        u1 = Uinp + (dU * self.space.cfg.dt * 0.5 / sweepc)

        ## TVD
        # right moving waves
        fr = (u1 * c + w) / 2
        fr_left = self.space.data_in_dir(fr.clone(), leftidx)
        dfl = (fr - fr_left) / 2
        dfr = self.space.data_in_dir(dfl.clone(), rightidx)
        fr = vanleer(fr, dfl, dfr)

        # left moving waves
        fl_curr = (u1 * c - w) / 2
        fl = self.space.data_in_dir(fl_curr.clone(), rightidx)
        fl_left = self.space.data_in_dir(fl.clone(), leftidx)
        dfl = (fl_left - fl) / 2
        ## go back to center cell, then go 2 cells to the right
        dfr = self.space.data_in_dir(dfl.clone(), rightidx)
        fl = vanleer(fl, dfl, dfr)

        dF = fr - fl

        # Take value from left cell
        dF_neighbor = self.space.data_in_dir(dF.clone(), leftidx)

        dFdx = (dF - dF_neighbor) / self.space.dx[axis]

        # suppress effect of periodic boundary conditions
        if not self.space.cfg.spherical:
            dFdx[:, self.space.boundary] = 0

        return dFdx

    def sweepx(self, U, sweepc):
        dFdx = self.calc_F(U, 0, sweepc)
        dU = self.external_torch - dFdx
        return U + (dU * self.space.cfg.dt / sweepc)

    def sweepy(self, U, sweepc):
        U_permuted = U[[0, 2, 1, 3]]
        dFdx = self.calc_F(U_permuted, 1, sweepc)
        dFdx = dFdx[[0, 2, 1, 3]]
        dU = self.external_torch - dFdx
        return U + (dU * self.space.cfg.dt / sweepc)

    def calc_next_U(self, U):

        Unext = U.clone()

        # Sweep count
        sweepc = 2

        Unext = self.sweepx(Unext, sweepc)
        Unext = self.sweepy(Unext, sweepc)
        Unext = self.sweepy(Unext, sweepc)
        Unext = self.sweepx(Unext, sweepc)

        if torch.isnan(Unext).any() or torch.isinf(Unext).any():
            print("Invalid values detected in Unext")

        return Unext

    def compute_cfl(self, U):
        velocities = torch.abs(U[1:3] / U[0]).sum(0)  # Fluid velocity

        # Calculate pressure using epsilon (internal energy)
        epsilon = U[3] - 0.5 * (U[0] * velocities ** 2)
        P = (self.space.cfg.gamma - 1) * epsilon
        P = torch.maximum(P, torch.zeros_like(P))

        # Speed of sound calculation
        sound_speed = torch.maximum(
            torch.sqrt(self.space.cfg.gamma * P / U[0]), torch.tensor(1e-5, dtype=U.dtype, device=U.device)
        )

        max_dx = max(self.space.cfg.dx)
        min_dx = min(self.space.cfg.dx)

        cfl = (velocities + sound_speed) * self.space.cfg.dt / max_dx

        # calculate valid timestep
        valid_dt = min_dx / (velocities + sound_speed)

        return cfl.max().item(), valid_dt.min().item()

    def run(self, target_time, start_time=0, checkpoint_freq=1, verbose=True, no_iter=False, auto_count=50):

        logger = logging.getLogger(__name__)

        if checkpoint_freq == "auto":
            _, dt = self.compute_cfl(self.U)
            iterc = target_time / dt
            checkpoint_freq = max(int(iterc / auto_count), 1)

            logger.info(f"Set checkpoint frequency to {checkpoint_freq}")

        # Adjust start time if any
        assert start_time < target_time
        self.all_times[0] += start_time

        highest_egy = torch.sum(self.U[3]).item()
        low_mass = max(torch.min(self.U[0]).item(), 1e-5)
        highest_v = max(torch.sqrt(torch.tensor(2 * highest_egy / low_mass)), 1e-20)
        smallest_dt = (
            min(self.space.cfg.dx) / highest_v
        )
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

            if torch.isnan(valid_dt):
                break

            total_time += self.space.cfg.dt
            t_steps += 1

            self.U = self.calc_next_U(self.U)

            info_txt = f"iter: {i}, dt: {self.space.cfg.dt}, total: {total_time}"

            if i % checkpoint_freq == 0:
                self.all_states.append(self.U)
                self.all_times.append(total_time)
                info_txt += " [Saved]"

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

        # Convert all_states to NumPy arrays
        all_states_np = [U.cpu().numpy() for U in self.all_states]

        return np.array(all_states_np), np.array(self.all_times)
