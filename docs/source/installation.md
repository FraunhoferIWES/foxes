# Installation

## Requirements

The supported Python versions are: 

- `Python 3.7`
- `Python 3.8`
- `Python 3.9`
- `Python 3.10`

## Virtual Python environment

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

## Standard users

As a standard user, you can install the latest release via [pip](https://pypi.org/project/foxes/) by
```console
pip install foxes
```
This in general corresponds to the `main` branch at [github](https://github.com/FraunhoferIWES/foxes). Alternatively, you can decide to install the latest pre-release developments (non-stable) by
```console
pip install git+https://github.com/FraunhoferIWES/foxes@dev#egg=foxes
```

## Developers

The first step as a developer is to clone the `foxes` repository by
```console
git clone https://github.com/FraunhoferIWES/foxes.git
```
Enter the root directory by `cd foxes`. Then you can either install from this directory via
```console
pip install -e .
```
Alternatively, add the `foxes` directory to your `PYTHONPATH`, e.g. by running
```console
export PYTHONPATH=`pwd`:$PYTHONPATH
```
from the root `foxes` directory, and then
```console
pip install -r requirements.txt
```