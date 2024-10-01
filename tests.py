### Basic imports
import numpy as np
import matplotlib.pyplot as plt
from os import path
import os
import glob
import logging
import sys
from cProfile import Profile
from pstats import Stats, SortKey

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
    if dist_to_center <= 0.2:
        return max(max_depth-3, 5)
    if dist_to_center <= 0.3:
        return max(max_depth-1, 5)
    elif dist_to_center <= 0.45:
        return max_depth
    elif dist_to_center <= 0.5:
        return max(max_depth-3, 5)
    else: 
        return min_depth

def gen_quadtree(width, height, min_depth, max_depth, max_resolution, files=None):
    logging.info(
        f"Generating quadtree with min_depth={min_depth}, max_depth={max_depth}, max_resolution={max_resolution}"
    )
    qt = Tree(width, height, min_depth, max_depth, depth_formula=depth_formula)

    logging.info("Generating helper constructs")
    # Generate nodes and neighbors
    c_l, n, v = qt.generate_all_nodes()
    logging.info(f"Leaf nodes: {len(c_l)}, Virtual nodes: {len(v)}")

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

    logging.info("Writing nodes to file")
    write_nodes_to_file(c_l, n, v, files[0])
    logging.info("Creating and writing square grid map to file")
    write_sqmatrix_to_file(qt, c_l, max_resolution, files[1])
    logging.info("Creating and writing boundary cell map to file")
    write_boundaries_to_file(qt, c_l, max_resolution, files[2])
    logging.info("Quadtree generation completed")

## 2D setup configs source:
"""
AVGERINOS, S., & RUSSO, G. (2019). 
All-Mach Number Solvers for Euler Equations 
and Saint-Venant-Exner Model.
(Chapter 8)
"""

def sedov_taylor_blast(res):
    s = np.array(res)  # space resolution
    target_time = 1  # time for final state
    gamma = 1.4
    energy_explosion = 0.979264 
    volume = np.array([4, 4])/np.sqrt(2)
    explosion_area = (volume / s).prod()
    logging.info(f"Explosion area: {explosion_area}")

    # ambient density
    density = np.ones(s) * 1

    p_init = (gamma - 1) * energy_explosion / explosion_area
    # ambient pressure
    ambient_energy = 1e-12
    pressure = np.ones(s) * ambient_energy * (gamma - 1)
    pressure[tuple(s//2)] = p_init

    logging.info(f"Total cells: {np.prod(s)}")

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

def range_incl(start,stop,step):
    n = start

    while n < stop-1:
        yield n
        n += step
    
    yield stop-1

def main():
    # Output directory
    output_dir = "simulation_outputs"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")


    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s:%(module)s:%(funcName)s | %(message)s', filename=f"{output_dir}/log.log")



    width, height = 1, 1  # dimensions of entire area
    min_depth = 1

    start_depth = 5
    end_depth = 13
    step = 64
    step2 = 256
    step3 = 1024

    # Resolutions for StdRes
    std_resolutions =  list(range(2**start_depth, 2**8+1, step)) +  list(range(2**8, 2**11+1, step2)) + list(range_incl(2**11, 2**end_depth+1, step3))
    # std_resolutions = range(0) # no iter

    # Resolutions for Quadtree (powers of 2)
    quadtree_resolutions = [2**i for i in range(start_depth, end_depth+1)]
    # quadtree_resolutions = range(0) # no iter

    do_no_iter = False


    # Initialize lists to store profiling times
    std_times = []
    quadtree_times = []

    # Run simulations for StdRes
    logging.info("Starting simulations for StdRes")

    with open(f"{output_dir}/sim_info.txt","w+"): pass

    with open(f"{output_dir}/sim_info.txt","a") as f:
        f.write("STDRES\nres total_time\n")

    for res in std_resolutions:
        logging.info(f"Running StdRes simulation at resolution {res}")
        cfg, target_time = sedov_taylor_blast((res, res))

        # Setup space
        space = StdRes(cfg)

        # Initial variables
        simulation = Simulation(space)

        profile_filename = path.join(output_dir, f"stdres_res{res}_profile.prof")
        with Profile() as profile:
            # Run simulation
            all_states, all_times = simulation.run(target_time, checkpoint_freq=200, verbose=True, no_iter=do_no_iter)
            profile.dump_stats(profile_filename)
            stats = Stats(profile)
            total_time = stats.total_tt
            std_times.append(total_time)
            logging.info(f"Simulation at resolution {res} completed in {total_time:.2f} seconds")
            logging.info(f"Profiler stats saved to {profile_filename}")

        # Convert all states to grid format
        logging.info(f"Converting states to grid format for resolution {res}")
        converted_states = [space.to_grid(state) for state in all_states]

        # Save converted states
        save_name = path.join(output_dir, f"stdres_res{res}.npz")
        np.savez_compressed(save_name, all_states=converted_states, all_times=all_times)
        logging.info(f"Simulation results saved to {save_name}")

        with open(f"{output_dir}/sim_info.txt","a") as f:
            f.write(f"{res} {total_time}\n")

    # Run simulations for Quadtree
    logging.info("Starting simulations for Quadtree")

    with open(f"{output_dir}/sim_info.txt","a") as f:
        f.write("QUADTREE\nres total_time\n")

    for res in quadtree_resolutions:
        logging.info(f"Running Quadtree simulation at resolution {res}")
        max_resolution = res
        # Generate Quadtree
        qtd_dir = f"Qtdata_res{res}"
        if not os.path.exists(qtd_dir):
            os.makedirs(qtd_dir)
        gen_quadtree(width, height, min_depth, int(np.log2(max_resolution)), max_resolution, files=qtd_dir)

        cfg, target_time = sedov_taylor_blast((max_resolution, max_resolution))

        # Setup space
        space = Quadtree(cfg, file_path=qtd_dir)

        # Initial variables
        simulation = Simulation(space)

        profile_filename = path.join(output_dir, f"quadtree_res{res}_profile.prof")
        with Profile() as profile:
            # Run simulation
            all_states, all_times = simulation.run(target_time, checkpoint_freq=200, verbose=True, no_iter=do_no_iter)
            profile.dump_stats(profile_filename)
            stats = Stats(profile)
            total_time = stats.total_tt
            quadtree_times.append(total_time)
            logging.info(f"Simulation at resolution {res} completed in {total_time:.2f} seconds")
            logging.info(f"Profiler stats saved to {profile_filename}")

        with open(f"{output_dir}/sim_info.txt","a") as f:
            f.write(f"{res} {total_time}\n")

        # Convert all states to grid format
        logging.info(f"Converting states to grid format for resolution {res}")
        converted_states = [space.to_grid(state) for state in all_states]

        # Save converted states
        save_name = path.join(output_dir, f"quadtree_res{res}.npz")
        np.savez_compressed(save_name, all_states=converted_states, all_times=all_times)
        logging.info(f"Simulation results saved to {save_name}")

    # Save profiling times
    profiling_times_file = path.join(output_dir, "profiling_times.npz")
    np.savez_compressed(profiling_times_file, std_res=std_resolutions, std_times=std_times, quadtree_res=quadtree_resolutions, quadtree_times=quadtree_times)
    logging.info(f"Profiling times saved to {profiling_times_file}")

if __name__ == "__main__":
    main()
