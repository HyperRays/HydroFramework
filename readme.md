# HydroFramework

This is an implementation of the framework described in the Matura Thesis of S. K. and D. I.  
If you are reading this offline, a GitHub repo is available at: [https://github.com/HyperRays/HydroFramework](https://github.com/HyperRays/HydroFramework)

## Framework Structure

There are three parts to the framework:

- **Configuration** (`config.py`)
- **Simulation** (`sim_<simulation_type>`)
- **Simulation Space** (`sim_<simulation_space>`, base class: `space.py`)

All these are composed and defined in the `tests.py` file.

## Getting Started

To get started, install the dependencies from `requirements.txt` using the command:

```bash
pip install -r requirements.txt
```

If you want to use torch simulations, please follow the instructions on the [PyTorch website](https://pytorch.org) to install it.

## Saves

Saves of various tests are provided in the `saves` folder. Please note that, other than `tests_final.7z`, none of the other saved tests are defined in `tests.py`.

To visualize the data, use a tool like Jupyter to open `LoadSave.ipynb`.

## Terminology

- `Quad`, `Q...`, or `tree` refer to `Quadtree`
- `StdRes` or `Homogeneous Resolution Space` refer to a `Regular Grid`
- `blast`, `sedov`, or `taylor` refer to the `Sedov-Taylor Blast` test
- `tube` or `sod` refer to the `Sod Shock Tube` test

## Description of Files

- **`config.py`**: Contains a class to define any configuration.
- **`gen_quadtree.py`**: Contains classes for a Quadtree and helper functions to flatten it.
- **`LoadSave.ipynb`**: Interactive Python notebook to visualize outputted data from the simulation runs.
- **`quadtree_helper.py`**: Contains helper functions to load a flattened Quadtree.
- **`sim_grid.py`**: Implementation of a regular grid space.
- **`sim_quadtree.py`**: Implementation of a quadtree space.
- **`sim_upwind_simple.py`**: First-order implementation of Upwind in NumPy.
- **`sim_upwind_simple_torch.py`**: PyTorch implementation of `sim_upwind_simple.py`.
- **`sim_upwind_TVD.py`**: Second-order implementation of Upwind using TVD in NumPy.
- **`sim_upwind_TVD_torch.py`**: PyTorch implementation of `sim_TVD_simple.py`.
- **`space.py`**: Contains a base class for any space.
- **`tests.py`**: Contains procedures for all test runs.

## Description of `tests.py`

The following tests are defined in `tests.py`:

| Test   | Space Type  | Simulation Type | Depth Function (only quadtree) |
|--------|-------------|-----------------|--------------------------------|
| blast  | stdres      | no tvd           |                                |
|        |             | tvd             |                                |
|        | quadtree    | no tvd           | sedov                          |
|        |             | tvd             | sedov                          |
| tube   | stdres      | no tvd           |                                |
|        |             | tvd             |                                |
|        | quadtree    | no tvd           | sod                            |
|        |             |                 | sod 2                          |
|        |             |                 | sod 3                          |
|        |             | tvd             | sod                            |
|        |             |                 | sod 2                          |
|        |             |                 | sod 3                          |

