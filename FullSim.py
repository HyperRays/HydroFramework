### Basic imports
import numpy as np
import matplotlib.pyplot as plt
from os import path

### component imports
from config import Config
from sim_quadtree import Quadtree
from sim_grid import StdRes
from sim_upwind_simple import Simulation
from gen_quadtree import (
    Tree,
    write_nodes_to_file,
    write_boundaries_to_file,
    write_sqmatrix_to_file,
)

def depth_formula(x, y, max_depth, min_depth, center_x=0.5, center_y=0.5):
    # default is proximity to center
    dist_to_center = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
    if dist_to_center <= 0.5:
        return max_depth
    else: 
        return min_depth

def gen_quadtree(width, height, min_depth, max_depth, max_resolution, files=None):
    print(
        f"▷ Generating tree with:\n└─min depth: {min_depth}\n└─max depth: {max_depth}\n└─max resolution {max_resolution}"
    )
    qt = Tree(width, height, min_depth, max_depth, depth_formula=depth_formula)

    print("▷ Generating helper constructs")
    # Generate nodes and neighbors
    c_l, n, v = qt.generate_all_nodes()
    print(f"▷ Info:\n   ◎{len(c_l)} leaf nodes\n   ◎{len(v)} virtual nodes")

    if isinstance(files, str):
        files = [
            path.join(files, "nodes_data.txt"),
            path.join(files, "sqmatrix_data.txt"),
            path.join(files, "boundary_data.txt"),
        ]
    elif isinstance(files, dict):
        files = [
            files["boundary"],
            files["nodes"],
            files["sqmap"],
        ]
    else:
        files = [
            "nodes_data.txt",
            "sqmatrix_data.txt",
            "boundary_data.txt",
        ]

    print("▷ Writing nodes to file")
    write_nodes_to_file(c_l, n, v, files[0])
    print("▷ Creating and writing square grid map to file")
    write_sqmatrix_to_file(qt, c_l, max_resolution, files[1])
    print("▷ Creating and writing boundary cell map to file")
    write_boundaries_to_file(qt, c_l, max_resolution, files[2])
    print("Done")

## 2D setup configs source:
"""
AVGERINOS, S., & RUSSO, G. (2019). 
All-Mach Number Solvers for Euler Equations 
and Saint-Venant-Exner Model.
(Chapter 8)
"""

# Initialize configuration and initial values for sod shock tube
def sod_shock_tube(res):
    # https://doi.org/10.1007/s10915-015-0134-0 -> Paper
    target_time = 0.2  # time for final state Wiki
    target_time = 0.168  # time for final state Avgerinos (2019)

    s = np.array(res)  # space resolution
    L = np.triu_indices(s[0])
    U_mask = np.zeros(s, dtype=bool)
    U_mask[:] = True
    U_mask[L] = False

    density = np.ones(s)
    density[U_mask] = 0.125

    pressure = np.ones(s)
    pressure[U_mask] = 0.1

    gamma = 1.4
    energy = pressure / (gamma - 1)

    cfg = Config(
        vol=np.sqrt([0.5, 0.5]),
        steps=s,
        list_p=density,
        list_v=np.zeros((2, *s)),  # contains both x and y
        list_e=energy,
        dt=0.0001,
        gamma=gamma,
        spherical=False,
    )

    return cfg, target_time

import numpy as np

def sedov_taylor_blast(res):
    
    s = np.array(res)  # space resolution
    target_time = 1  # time for final state
    gamma = 1.4
    energy_explosion = 0.979264 
    volume = np.array([4, 4])/np.sqrt(2)
    explosion_area = (volume / s).prod()
    # explosion_radius = 1 / 100
    print(f"explosion area: {explosion_area}")

    # quit()

    # ambient density
    density = np.ones(s) * 1


    p_init = (gamma - 1) * energy_explosion / explosion_area
    # ambient pressure
    ambient_energy = 1e-12
    pressure = np.ones(s) * ambient_energy * (gamma - 1)
    # print(pressure[int(s[0]) // 2, int(s[1]) // 2])

    pressure[tuple(s//2)] = p_init

    # print(f"Filled Cell count: {cell_count}")
    print(f"Cell count {np.prod(s)}")

    energy = pressure / (gamma - 1)

    cfg = Config(
        vol=volume,
        steps=s,
        list_p=density,
        list_v=np.zeros((2,*s)), # contains both x and y
        list_e=energy,
        dt=0.0001,
        gamma=gamma,
        spherical=False
    )

    return cfg, target_time


def EOS(res, 
        rl, ul, pl,
        rr, ur, pr, 
        t, gamma):

    target_time = t

    s = np.array(res)  # space resolution
    L = np.triu_indices(s[0])
    U_mask = np.zeros(s, dtype=bool)
    U_mask[:] = True
    U_mask[L] = False

    density = np.ones(s) * rr
    density[U_mask] = rl

    if ul != 0:
        ul = ul/np.sqrt(2)

    if ur != 0:
        ul = ul/np.sqrt(2)
    
    velocity = np.ones((2,*s)) * ur
    velocity[:,U_mask] = ul

    pressure = np.ones(s) * pr
    pressure[U_mask] = pl

    energy = pressure / (gamma - 1)

    cfg = Config(
        vol=np.sqrt([0.5, 0.5]),
        steps=s,
        list_p=density,
        list_v=velocity,  # contains both x and y
        list_e=energy,
        dt=0.0001,
        gamma=gamma,
        spherical=False,
    )

    return cfg, target_time

def main():
    width, height = 1, 1  # dimensions of entire area
    min_depth = 1  # maximum depth of quadtree at edges
    max_depth = 5 # maximum depth of quadtree at center
    max_resolution = 2**max_depth

    # gen_quadtree(width, height, min_depth, max_depth, max_resolution)

    times = []
    # ,125,128,150,200,250
    for res in map(lambda v: int((v+1)*(512-254)*0.1) + 254, range(10)):
        print(res)
        max_resolution = res
        cfg, target_time = sedov_taylor_blast((max_resolution,max_resolution))
        # return
        # cfg, target_time = EOS((max_resolution,max_resolution),
        #                         rl=3.857143, ul=-0.810631, pl=31./3.,
        #                          rr=1.,       ur=-3.44,     pr=1.,
        #                         t=1, gamma=1.4)

        # setup space
        space = StdRes(cfg)
        # space = Quadtree(cfg)

        # Initial variables
        simulation = Simulation(space)


        from cProfile import Profile
        from pstats import SortKey, Stats

        with Profile() as profile:
            # run simulation and convert space back to regular grid
            all_states, all_times = simulation.run(target_time, checkpoint_freq=200, verbose=False)
            stats = Stats(profile)
            # (
            #     stats
            #     .strip_dirs()
            #     .sort_stats(SortKey.TIME)
            #     .print_stats()
            # )

            times += [stats.total_tt]
    
    print(times)

    # tmp = []

    # for idx in range(len(all_times)):
    #     tmp += [space.to_grid(all_states[idx])]

    # all_states = np.array(tmp)

    # np.savez_compressed("save_eos.npz",**{"all_states":all_states, "all_times": all_times})
    



if __name__ == "__main__":
    main()