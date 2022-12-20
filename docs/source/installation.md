# Installation

## Requirements

The supported Python versions are:

- `Python 3.7`
- `Python 3.8`
- `Python 3.9`
- `Python 3.10`

## Installation via conda

### Virtual Python environment

First create a new `conda` environment, for example called `foxes`, by

```console
conda create --name foxes
```

Then activate the environment every time you work with `foxes`, by

```console
conda activate foxes
```

You can leave the environment by

```console
conda deactivate
```

The `conda` installation commands below should be executed within the active `foxes` environment.

### Standard users

The `foxes` package is available on the channel [conda-forge](https://anaconda.org/conda-forge/foxes). You can install the latest version by

```console
conda install -c conda-forge foxes
```

### Developers

For developers using `conda`, we recommend first installing foxes, then removing only the `foxes` package while keeping the dependencies, and then adding `foxes` again from a git using `conda develop`:

```console
conda install -c conda-forge foxes
conda remove foxes --force
git clone https://github.com/FraunhoferIWES/foxes.git
cd foxes
conda develop .
```
The last line makes sure that all your code changes are included whenever importing `foxes`.

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

All subsequent installation commands via `pip` can then be executed directly within the active environment without changes. After your work with `foxes` is done you can leave the environment by the command 

```console
deactivate
``` 

### Standard users

As a standard user, you can install the latest release via [pip](https://pypi.org/project/foxes/) by

```console
pip install foxes
```

This commands installs the version that correspond to the `main` branch at [github](https://github.com/FraunhoferIWES/foxes). Alternatively, you can decide to install the latest pre-release developments (non-stable) by

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

Then you can then install from this directory, following all your code changes, via

```console
pip install -e .
```
