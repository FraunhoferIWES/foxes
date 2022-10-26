# Installation

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
