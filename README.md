# foxes
## Farm Optimization and eXtended yield Evaluation Software

![](docs/logo/Logo_FOXES_IWES.svg)

## Overview
The software `foxes` is a wind farm simulation and wake modelling tool which is based on engineering wake models. It has many applications, for example
- Wind farm optimization, e.g. layout optimization or wake steering,
- Wind farm post-construction analysis,
- Wake model studies, comparison and validation,
- Wind farm simulations invoking complex model chains.

Currently the modelled time scales are related to 10-min averages or longer periods, and also statistical data like wind rose data can be modelled. High-frequency effects are not supported so far.

## Installation
- We recommend working in a Python virtual environment and install `foxes` there. Such an environment can be created by
```
python -m venv /path/to/my_venv
```
and afterwards be activated by
```
source /path/to/my_venv/bin/activate
```
You can leave the environment by the command `deactivate`.
- As a general user, you can install the latest release by
```
pip install foxes
```
This should correspond to the `main` branch here at GitHub.
- As a devloper, you can either install from this directory via
```
pip install -e .
```
- Alternatively, you can add the path to your local `foxes` clone directory to your `PYTHONPATH`, e.g. by
```
export PYTHONPATH=`pwd`:$PYTHONPATH
```
and then run
```
pip install -r requirements.txt
```

## Minimal example

For detailed examples, check the `examples` folder in this repository. A minimal running example is the following, based on provided static `csv` data files:
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