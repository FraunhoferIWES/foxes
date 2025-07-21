# Installation

## Requirements

The supported Python versions are `Python 3.9`...`3.13`.

## TLDR; Quick installation guide

Either install via pip:

```console
pip install foxes
```

Alternatively, install via conda:

```console
conda install foxes -c conda-forge
```

More details, including guidelines for developers and 
quicker conda installations, can be found below.

## Installation via pip

### Virtual Python environment

First create a new `venv` environment, for example called `foxes` and located at `~/venv/foxes` (choose any other convenient name and location in your file system if you prefer), by

```console
python3 -m venv ~/venv/foxes
```

Then activate the environment every time you work with `foxes`, by

```console
source ~/venv/foxes/bin/activate
```

You can leave the environment by

```console
deactivate
```

The `pip` installation commands below should be executed within the active `foxes` environment.

### Standard users

As a standard user, you can install the latest release via [pip](https://pypi.org/project/foxes/) by

```console
pip install foxes
```

This commands installs the version that correspond to the `main` branch at [github](https://github.com/FraunhoferIWES/foxes). Alternatively, you can decide to install the latest pre-release developments (non-stable) by

```console
pip install git+https://github.com/FraunhoferIWES/foxes@dev
```

If you are planning to run wind farm optimizations, please install the 
[foxes-opt](https://github.com/FraunhoferIWES/foxes-opt) package:

```console
pip install foxes[opt]
```

or

```console
pip install foxes-opt
```

### Developers

For developers using `pip`, simply invoke the `-e` flag in the installation command in your local clone:

```console
git clone https://github.com/FraunhoferIWES/foxes.git
cd foxes
pip install -e .
```
The last line makes sure that all your code changes are included whenever importing `foxes`. Concerning the `git clone` line, we actually recommend that you fork `foxes` on GitHub and then replace that command by cloning your fork instead.

If you are planning to run wind farm optimizations, please also install the 
[foxes-opt](https://github.com/FraunhoferIWES/foxes-opt) package as described above.

### Optional dependencies

The following optional dependencies are available:


| Option | Usage                              |
|--------|------------------------------------|
| opt    | Installs [foxes-opt](https://github.com/FraunhoferIWES/foxes-opt)  |
| dask   | Installs dependencies for `dask` engine |
| test   | Dependencies for running the tests |
| doc    | Dependencies for creating the docs |
| utils  | Dependencies for utilities         |

As an example, the optional dependencies `test` are installed by

```console
pip install foxes[test]
```

or, for development mode, from the `test` root directory by

```console
pip install -e .[test]
```

Note that options can also be combined, e.g.

```console
pip install foxes[test,opt]
```

## Installation via conda

### Preparation (optional)

It is recommend to use the `libmamba` dependency solver instead of the default solver. Install it once by

```console
conda install conda-libmamba-solver -n base -c conda-forge
```

We recommend that you set this to be your default solver, by

```console
conda config --set solver libmamba
```

### Virtual Python environment

First create a new `conda` environment, for example called `foxes`, by

```console
conda create -n foxes -c conda-forge
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
conda install foxes -c conda-forge --solver=libmamba
```

If you are planning to run wind farm optimizations, please install the 
[foxes-opt](https://github.com/FraunhoferIWES/foxes-opt) package instead:

```console
conda install foxes-opt -c conda-forge --solver=libmamba
```

Note that the `--solver=libmamba` in both above commands is optional. Note that it is not necessary if you have set the `libmamba` solver as your default, see above.

### Developers

*The following steps require Python >= 3.11*

For developers using `conda`, we recommend first installing foxes as described above, then removing only the `foxes` package while keeping the dependencies, and then adding `foxes` again from a git using `conda develop`:

```console
conda install foxes conda-build -c conda-forge --solver=libmamba
conda remove foxes --force
git clone https://github.com/FraunhoferIWES/foxes.git
cd foxes
conda develop .
```

The last line makes sure that all your code changes are included whenever importing `foxes`. 
Concerning the `git clone` line, we actually recommend that you fork `foxes` on GitHub and then replace that command by cloning your fork instead.

If you are planning to run wind farm optimizations, please install the 
[foxes-opt](https://github.com/FraunhoferIWES/foxes-opt) package in addition:

```console
conda install foxes-opt -c conda-forge --solver=libmamba
```

The `--solver=libmamba` is optional. Note that it is not necessary if you have set the `libmamba` solver as your default, see above.
