# Welcome to foxes

![FOXES Logo](Logo_FOXES.svg)

## Overview

The software `foxes` is a modular wind farm simulation and wake modelling toolbox which is based on engineering wake models. It has many applications, for example

- Wind farm optimization, e.g. layout optimization or wake steering,
- Wind farm post-construction analysis,
- Wake model studies, comparison and validation,
- Wind farm simulations invoking complex model chains.

The calculation is fully vectorized and its fast performance is owed to [dask](https://www.dask.org/). Also the parallelization on local or remote clusters is enabled via `dask`. The wind farm
optimization capabilities invoke the [iwopy](https://github.com/FraunhoferIWES/iwopy) package which
as well supports vectorization.

`foxes` is build upon many years of experience with wake model code development at IWES, starting with the C++ based in-house code _flapFOAM_ (2011-2019) and the Python based direct predecessor _flappy_ (2019-2022).

Documentation: [https://fraunhoferiwes.github.io/foxes.docs/index.html](https://fraunhoferiwes.github.io/foxes.docs/index.html)

Source code: [https://github.com/FraunhoferIWES/foxes](https://github.com/FraunhoferIWES/foxes)

PyPi reference: [https://pypi.org/project/foxes/](https://pypi.org/project/foxes/)

## Requirements

The supported Python versions are: 

- `Python 3.7`
- `Python 3.8`
- `Python 3.9`
- `Python 3.10`

## Installation

### Virtual Python environment

We recommend working in a Python virtual environment and install `foxes` there. Such an environment can be created by

```console
python -m venv /path/to/my_venv
```

and afterwards be activated by

```console
source /path/to/my_venv/bin/activate
```

Note that in the above commands `/path/to/my_venv` is a placeholder that should be replaced by a path to a (non-existing) folder of your choice, for example `~/venv/foxes`.

All subsequent installation commands via `pip` can then be executed directly within the active environment without changes. After your work with `foxes` is done you can leave the environment by the command `deactivate`. 

### Standard users

As a standard user, you can install the latest release via [pip](https://pypi.org/project/foxes/) by

```console
pip install foxes
```

If you are also interested in running wind farm optimizations, please install

```console
pip install foxes[opt]
```

Both commands install versions that correspond to the `main` branch at [github](https://github.com/FraunhoferIWES/foxes). Alternatively, you can decide to install the latest pre-release developments (non-stable) by

```console
pip install git+https://github.com/FraunhoferIWES/foxes@dev#egg=foxes
```

### Developers

The first step as a developer is to clone the `foxes` repository by

```console
git clone https://github.com/FraunhoferIWES/foxes.git
```

Enter the root directory by `cd foxes`. Then you can either install from this directory via

```console
pip install -e .
```

or if you are also interested in running wind farm optimizations, then

```console
pip install -e .[opt]
```

Alternatively, add the `foxes` directory to your `PYTHONPATH`, e.g. by running

```console
export PYTHONPATH=`pwd`:$PYTHONPATH
```

from the root `foxes` directory, and then

```console
pip install -r requirements.txt
```

For running optimizations, please install in addition

```console
pip install iwopy pymoo
```

## Usage

For detailed examples of how to run _foxes_, check the `examples` and `notebooks` folders in this repository. A minimal running example is the following, based on provided static `csv` data files:

```python
import foxes

states = foxes.input.states.Timeseries("timeseries_3000.csv.gz", ["WS", "WD","TI","RHO"])

mbook = foxes.ModelBook("NREL-5MW-D126-H90.csv")

farm = foxes.WindFarm()
foxes.input.farm_layout.add_from_file(farm,"test_farm_67.csv",turbine_models=["Pct"])

algo = foxes.algorithms.Downwind(mbook, farm, states, ["Jensen_linear_k007"])
farm_results = algo.calc_farm()

print(farm_results)
```

## Acknowledgements

The development of _foxes_ and its predecessors _flapFOAM_ and _flappy_ (internal - non public) has been supported through multiple publicly funded research projects. We acknowledge in particular the funding by the Federal Ministry of Economic Affairs and Climate Action (BMWK) through the projects _Smart Wind Farms_ (grant no. 0325851B), _GW-Wakes_ (0325397B) and _X-Wakes_ (03EE3008A) as well as the funding by the Federal Ministry of Education and Research (BMBF) in the framework of the project _H2Digital_ (03SF0635).
