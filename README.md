# foxes
## Farm Optimization and eXtended yield Evaluation Software

![](Logo_FOXES_IWES.svg)

## Overview
The software `foxes` is a modular wind farm simulation and wake modelling toolbox which is based on engineering wake models. It has many applications, for example
- Wind farm optimization, e.g. layout optimization or wake steering,
- Wind farm post-construction analysis,
- Wake model studies, comparison and validation,
- Wind farm simulations invoking complex model chains.

Currently the modelled time scales are related to 10-min averages or longer periods, and also statistical data like wind rose data can be modelled. High-frequency effects are not supported.

Source code: [https://github.com/FraunhoferIWES/foxes](https://github.com/FraunhoferIWES/foxes)
PyPi reference: [https://pypi.org/project/foxes/](https://pypi.org/project/foxes/)

## Requirements
The installation requires `Python >= 3.7`.

## Installation

### Virtual Python environment

We recommend working in a Python virtual environment and install `foxes` there. Such an environment can be created by
```
python -m venv /path/to/my_venv
```
and afterwards be activated by
```
source /path/to/my_venv/bin/activate
```
All subsequent installation commands via `pip` can then be executed directly within the active environment without changes. After your work with `foxes` is done you can leave the environment by the command `deactivate`. 

### Standard users

- As a standard user, you can install the latest release via [pip](https://pypi.org/project/foxes/) by
```
pip install foxes
```
This in general corresponds to the `main` branch at [github](https://github.com/FraunhoferIWES/foxes). Alternatively, you can decide to install the latest pre-release developments by
```
pip install git+https://github.com/FraunhoferIWES/foxes/tree/dev
```

### Developers

- The first step as a developer is to clone the `foxes` repository by
```
git clone https://github.com/FraunhoferIWES/foxes.git
```
Enter the root directory by `cd foxes`. Then you can either install from this directory via
```
pip install -e .
```
- Alternatively, add the `foxes` directory to your `PYTHONPATH`, e.g. by running
```
export PYTHONPATH=`pwd`:$PYTHONPATH
```
from the root `foxes` directory, and then
```
pip install -r requirements.txt
```

## Minimal example

For detailed examples of how to run _foxes_, check the `examples` and `notebooks` folders in this repository. A minimal running example is the following, based on provided static `csv` data files:
```python
import foxes

states = foxes.input.states.Timeseries("timeseries_3000.csv.gz", ["WS", "WD","TI","RHO"])

farm = foxes.WindFarm()
foxes.input.farm_layout.add_from_file(farm,"test_farm_67.csv",turbine_models=["Pct"])

mbook = foxes.ModelBook("NREL-5MW-D126-H90.csv")

algo = foxes.algorithms.Downwind(mbook, farm, states, ["Jensen_linear_k007"])
algo.calc_farm()

print(farm_results)
```