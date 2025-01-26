# pylint: disable=all
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

from sim_upwind_simple import Simulation as no_tvd_simulation
from sim_upwind_tvd import Simulation as tvd_simulation

# try:
#     import torch
#     torch.set_num_threads(24)

#     from sim_upwind_simple_torch import Simulation as no_tvd_simulation_torch
#     from sim_upwind_tvd_torch import Simulation as tvd_simulation_torch

# except:
#     print("torch not found")


from gen_quadtree import (
    Tree,
    write_nodes_to_file,
    write_boundaries_to_file,
    write_sqmatrix_to_file,
)

def depth_formula_sedov(x, y, max_depth, min_depth, center_x=0.5, center_y=0.5):
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

def depth_formula_sod(x, y, max_depth, min_depth, center_x=0.5, center_y=0.5):
    # Rotation matrix
    # | cos(θ) -sin(θ) |
    # | sin(θ)  cos(θ) |

    theta = np.pi * 1.25
    M = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    center = np.array([center_x, center_y])
    rot = (np.array([x, y]) - center) @ M + center 

    if rot[1] < 0.1:
        return min_depth
    if rot[1] < 0.55 and (0.6 >= rot[0] >= 0.4):
        return max_depth

    if 0.6 >= rot[0] >= 0.4:
        return max_depth

    if 0.67 >= rot[0] >= 0.33:
        return max_depth-2

    if rot[1] < 0.6:
        return max_depth-2
    
    return min_depth


def faulty1(x, y, max_depth, min_depth, center_x=0.5, center_y=0.5):
    # Rotation matrix
    # | cos(θ) -sin(θ) |
    # | sin(θ)  cos(θ) |

    theta = np.pi * 1.25
    M = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    center = np.array([center_x, center_y])
    rot = (np.array([x, y]) - center) @ M + center 
    # print(rot)
    if rot[1] < 0.1:
        return min_depth
    if rot[1] < 0.55:
        return max_depth

    if 0.6 >= rot[0] >= 0.4:
        return max_depth

    if 0.67 >= rot[0] >= 0.33:
        return max_depth-2

    if rot[1] < 0.6:
        return max_depth-2

    return min_depth


def faulty2(x, y, max_depth, min_depth, center_x=0.5, center_y=0.5):
    # Rotation matrix
    # | cos(θ) -sin(θ) |
    # | sin(θ)  cos(θ) |

    theta = np.pi * 1.25
    M = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    center = np.array([center_x, center_y])
    rot = (np.array([x, y]) - center) @ M + center 

    if 0.67 >= rot[0] >= 0.33:
        return max_depth-2

    return min_depth


def gen_quadtree(width, height, min_depth, max_depth, max_resolution, depth_func, files=None):
    logging.info(
        f"Generating quadtree with min_depth={min_depth}, max_depth={max_depth}, max_resolution={max_resolution}"
    )
    qt = Tree(width, height, min_depth, max_depth, depth_formula=depth_func)

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

def EOS(res, 
        rl, ul, pl,
        rr, ur, pr, 
        t, gamma):
    """
    create configurations for Equations Of State
    """

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

def sod_shock_tube(res):
    return EOS(res, rl=1.0,   ul=0.,   pl=1.0,
                    rr=0.125, ur=0.,   pr=0.1,
                    t = 0.168, gamma = 1.4)

def range_incl(start,stop,step):
    n = start

    while n < stop-1:
        yield n
        n += step
    
    yield stop-1

def run_std_res_tests(simulationf, resolutions, output_dir, do_no_iter, test, std_times, use_torch):

    # Run simulations for StdRes
    logging.info("Starting simulations for StdRes")

    with open(f"{output_dir}/sim_info.txt","a") as f:
        f.write("STDRES\nres total_time\n")

    for res in resolutions:
        logging.info(f"Running StdRes simulation at resolution {res}")
        cfg, target_time = test((res, res))

        # Setup space
        space = StdRes(cfg, use_torch=use_torch)

        # Initial variables
        simulation = simulationf(space)

        profile_filename = path.join(output_dir, f"stdres_res{res}_profile.prof")
        with Profile() as profile:
            # Run simulation
            all_states, all_times = simulation.run(target_time, checkpoint_freq=500, verbose=True, no_iter=do_no_iter)
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

def run_quad_res_tests(simulationf, resolutions, output_dir, do_no_iter, test, width, height, min_depth, quadtree_times, depth_func, use_torch):
    # Run simulations for Quadtree
    logging.info("Starting simulations for Quadtree")

    with open(f"{output_dir}/sim_info.txt","a") as f:
        f.write("QUADTREE\nres total_time\n")

    for res in resolutions:
        logging.info(f"Running Quadtree simulation at resolution {res}")
        max_resolution = res
        # Generate Quadtree
        qtd_dir = f"{output_dir}/Qtdata_res{res}"
        if not os.path.exists(qtd_dir):
            os.makedirs(qtd_dir)
        gen_quadtree(width, height, min_depth, int(np.log2(max_resolution)), max_resolution, depth_func, files=qtd_dir)

        cfg, target_time = test((max_resolution, max_resolution))

        # Setup space
        space = Quadtree(cfg, file_path=qtd_dir, use_torch=use_torch)

        # Initial variables
        simulation = simulationf(space)

        profile_filename = path.join(output_dir, f"quadtree_res{res}_profile.prof")
        with Profile() as profile:
            # Run simulation
            all_states, all_times = simulation.run(target_time, checkpoint_freq=500, verbose=True, no_iter=do_no_iter)
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

def main():
    # Tests done:
    # 32 -> 512
    # Blast:
    #   NO TVD/ TVD
    #       STDRES / QUADTREE
    # Tube:
    #   No TVD/TVD:
    #       STDRES
    #   No TVD/TVD:
    #       Quad: proper
    #       Quad: faulty1
    #       Quad: faulty2

    # Configure logging 
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s:%(module)s:%(funcName)s | %(message)s', filename=f"log.log")

    width, height = 1, 1  # dimensions of entire area
    min_depth = 1

    start_depth = 5
    end_depth = 9

    do_no_iter = True

    std_resolutions = [2**i for i in range(start_depth, end_depth+1)]
    quadtree_resolutions = [2**i for i in range(start_depth, end_depth+1)]
    

    if not os.path.exists("tests"):
        os.makedirs("tests")
        print(f"Created output directory: tests")

    combinations = [
        # folder, space-type, resolutions, tvd/no-tvd, test, depth_func, torch
        # Blast
        ## STDRES
        ["tests/std_blast_no_tvd", "std", std_resolutions, False, "blast", None, False],
        ["tests/std_blast_tvd", "std", std_resolutions, True, "blast", None, False],
        ## QUAD
        ["tests/quad_blast_no_tvd", "quad", quadtree_resolutions, False, "blast", depth_formula_sedov, False],
        ["tests/quad_blast_tvd", "quad", quadtree_resolutions, True, "blast", depth_formula_sedov, False],

        # Tube
        ## STDRES
        ["tests/std_tube_no_tvd", "std", std_resolutions, False, "tube", None, False],
        ["tests/std_tube_tvd", "std", std_resolutions, True, "tube", None, False],
        ## QUAD: proper
        ["tests/quad_tube_no_tvd_proper", "quad", quadtree_resolutions, False, "tube", depth_formula_sod, False],
        ["tests/quad_tube_tvd_proper", "quad", quadtree_resolutions, True, "tube", depth_formula_sod, False],
        ## QUAD: faulty1
        ["tests/quad_tube_no_tvd_faulty1", "quad", quadtree_resolutions, False, "tube", faulty1, False],
        ["tests/quad_tube_tvd_faulty1", "quad", quadtree_resolutions, True, "tube", faulty1, False],
        ## QUAD: faulty2
        ["tests/quad_tube_no_tvd_faulty2", "quad", quadtree_resolutions, False, "tube", faulty2, False],
        ["tests/quad_tube_tvd_faulty2", "quad", quadtree_resolutions, True, "tube", faulty2, False],
    ]


    for (folder, space_type, resolutions, tvd, test, depth_func, use_torch) in combinations:
        
        # Output directory
        output_dir = folder

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        times = []
        log_text = ""

        if test == "tube":
            testf = sod_shock_tube
            log_text += " Test: sod "
        elif test == "blast":
            testf = sedov_taylor_blast
            log_text += " Test: blast "

        if use_torch:
            if tvd:
                simulationf = tvd_simulation_torch
                log_text += " TVD: True "
            else:
                simulationf = no_tvd_simulation_torch
                log_text += " TVD: False "
        else:
            if tvd:
                simulationf = tvd_simulation
                log_text += " TVD: True "
            else:
                simulationf = no_tvd_simulation
                log_text += " TVD: False "

        if space_type == "std":
            runf_space = run_std_res_tests
            args = (simulationf, resolutions, output_dir, do_no_iter, testf, times, use_torch)
            log_text += " Space: Grid "

        elif space_type == "quad":
            runf_space = run_quad_res_tests
            args = (simulationf, resolutions, output_dir, do_no_iter, testf, width, height, min_depth, times, depth_func, use_torch)
            log_text += " Space: Quadtree "


        logging.info(log_text)

        #create file
        with open(f"{output_dir}/sim_info.txt","w+"): pass

        runf_space(*args)

        # Save profiling times
        profiling_times_file = path.join(output_dir, "profiling_times.npz")
        np.savez_compressed(profiling_times_file, res=resolutions, times=times)
        logging.info(f"Profiling times saved to {profiling_times_file}")


if __name__ == "__main__":
    main()