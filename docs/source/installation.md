# Installation

## Requirements

The supported Python versions are:

- `Python 3.7`
- `Python 3.8`
- `Python 3.9`
- `Python 3.10`

## Installation via conda

The `foxes` package is available on the channel [conda-forge](https://anaconda.org/conda-forge/foxes). You can install the latest version by

```console
conda install -c conda-forge foxes
```

## Installation via pip

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

Enter the root directory by 

```console
cd foxes
```

Then you can either install from this directory via

```console
pip install -e .
```

or if you are also interested in running wind farm optimizations, then

```console
pip install -e .[opt]
```
