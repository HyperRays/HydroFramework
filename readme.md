# HydroFramework

This is an implementation of the Framework described in the Matura Thesis of  S. K. and D. I.
If you are reading this offline, a github repo is available at: [https://github.com/HyperRays/HydroFramework](https://github.com/HyperRays/HydroFramework)

There are three parts to the framework:

- Configuration (`config.py`)
- Simulation (sim\___simulation type__)
- Simulation Space (sim\___simulation space__, base class:`space.py`)

All these are then composed and defined in the `tests.py` file.

To get started install the dependencies from `requirements.txt` using `pip install -r requirements.txt`
If you want to use torch simulations, please follow the instructions on the pytorch website to install it.

Saves of various tests are provided in the `saves` folder. Please note that other than `tests_final.7z` none of the other tests saved are defined in `tests.py`. 
If you want to visualize the data please use a tool like `jupyter` to open `LoadSave.ipynb`


### Terms

`Quad`,`Q...` or `tree` refer to `Quadtree` \
`StdRes` or `Homogeneous Resolution Space` refer to a `regular Grid` \
`blast`,`sedov` or `taylor` refer to the `sedov-taylor blast` test \
`tube` or `sod` refer to the `sod shock tube` test 

### Description of Files

`config.py`: Contains a class to define any configuration \ 
`gen_quadtree`: Contains classes for a Quadtree as well as helper functions to flatten it \
`LoadSave.ipynb`: Interactive python notebook to visualize the outputted data from the simulation runs \
`quadtree_helper.py`: Contains helper functions to load a flattened quadtree \
`sim_grid`: Implementation of a regular grid space \
`sim_quadtree`: Implementation of a quadtree space \
`sim_upwind_simple.py`: A first-order implementation of upwind in numpy \
`sim_upwind_simple_torch.py`: torch implementation of `sim_upwind_simple.py` \ 
`sim_upwind_TVD.py`: A second-order implementation of upwind using TVD in numpy \
`sim_upwind_TVD_torch.py`: torch implementation of `sim_TVD_simple.py` \
`space.py`: Contains a base class for any space \
`tests.py`: Contains the procedures for all test runs 

### Description of `test.py`

The following tests are defined in `test.py`:
| Test | Space Type | Simulation type | Depth Function (only quadtree) |
|-------|----------|--------|-------|
| blast | stdres   | no tvd |       |
|       |          | tvd    |       |
|       | quadtree | no tvd | sedov |
|       |          | tvd    | sedov |
| tube  | stdres   | no tvd |       |
|       |          | tvd    |       |
|       | quadtree | no tvd | sod   |
|       |          |        | sod 2 |
|       |          |        | sod 3 |
|       |          | tvd    | sod   |
|       |          |        | sod 2 |
|       |          |        | sod 3 |


