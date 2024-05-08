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

Anaconda reference: [https://anaconda.org/conda-forge/foxes](https://anaconda.org/conda-forge/foxes)

## Citation

Please cite the JOSS paper `"FOXES: Farm Optimization and eXtended yield
Evaluation Software"` 

 [![DOI](https://joss.theoj.org/papers/10.21105/joss.05464/status.svg)](https://doi.org/10.21105/joss.05464)

 Bibtex:
 ```
@article{
    Schmidt2023, 
    author = {Jonas Schmidt and Lukas Vollmer and Martin Dörenkämper and Bernhard Stoevesandt}, 
    title = {FOXES: Farm Optimization and eXtended yield Evaluation Software}, 
    doi = {10.21105/joss.05464}, 
    url = {https://doi.org/10.21105/joss.05464}, 
    year = {2023}, 
    publisher = {The Open Journal}, 
    volume = {8}, 
    number = {86}, 
    pages = {5464}, 
    journal = {Journal of Open Source Software} 
}
 ```

## Installation via pip

The supported Python versions are: 

- `Python 3.8`
- `Python 3.9`
- `Python 3.10`
- `Python 3.11`
- `Python 3.12`

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
pip install git+https://github.com/FraunhoferIWES/foxes@dev#egg=foxes
```

### Developers

For developers using `pip`, simply invoke the `-e` flag in the installation command in your local clone:

```console
git clone https://github.com/FraunhoferIWES/foxes.git
cd foxes
pip install -e .
```
The last line makes sure that all your code changes are included whenever importing `foxes`. Concerning the `git clone` line, we actually recommend that you fork `foxes` on GitHub and then replace that command by cloning your fork instead.

## Installation via conda

The supported Python versions are: 

- `Python 3.8`
- `Python 3.9`
- `Python 3.10`
- `Python 3.11`
- `Python 3.12`

### Preparation

It is strongly recommend to use the `libmamba` dependency solver instead of the default solver. Install it once by

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

The `--solver=libmamba` is optional. Note that it is not necessary if you have set the `libmamba` solver as your default, see above.

### Developers

For developers using `conda`, we recommend first installing foxes as described above, then removing only the `foxes` package while keeping the dependencies, and then adding `foxes` again from a git using `conda develop`:

```console
conda install foxes conda-build -c conda-forge --solver=libmamba
conda remove foxes --force
git clone https://github.com/FraunhoferIWES/foxes.git
cd foxes
conda develop .
```

The last line makes sure that all your code changes are included whenever importing `foxes`. The `--solver=libmamba` is optional. Note that it is not necessary if you have set the `libmamba` solver as your default, see above.

Concerning the `git clone` line, we actually recommend that you fork `foxes` on GitHub and then replace that command by cloning your fork instead.

## Usage

For detailed examples of how to run _foxes_, check the `examples` and `notebooks` folders in this repository. A minimal running example is the following, based on provided static `csv` data files:

```python
import foxes

states = foxes.input.states.Timeseries("timeseries_3000.csv.gz", ["WS", "WD","TI","RHO"])

farm = foxes.WindFarm()
foxes.input.farm_layout.add_from_file(farm, "test_farm_67.csv", turbine_models=["NREL5MW"])

algo = foxes.algorithms.Downwind(farm, states, ["Jensen_linear_k007"])
farm_results = algo.calc_farm()

print(farm_results)
```

## Testing

For testing, please clone the repository and install the required dependencies:
```console
git clone https://github.com/FraunhoferIWES/foxes.git
cd foxes
pip install -e .[test]
```

The tests are then run by
```console
pytest tests
```

## Contributing

1. Fork _foxes_ on _github_.
2. Create a branch (`git checkout -b new_branch`)
3. Commit your changes (`git commit -am "your awesome message"`)
4. Push to the branch (`git push origin new_branch`)
5. Create a pull request [here](https://github.com/FraunhoferIWES/foxes/pulls)

## Acknowledgements

The development of _foxes_ and its predecessors _flapFOAM_ and _flappy_ (internal - non public) has been supported through multiple publicly funded research projects. We acknowledge in particular the funding by the Federal Ministry of Economic Affairs and Climate Action (BMWK) through the projects _Smart Wind Farms_ (grant no. 0325851B), _GW-Wakes_ (0325397B) and _X-Wakes_ (03EE3008A), as well as the funding by the Federal Ministry of Education and Research (BMBF) in the framework of the project _H2Digital_ (03SF0635). We furthermore acknowledge funding by the Horizon Europe project FLOW (Atmospheric Flow, Loads and pOwer 
for Wind energy - grant id 101084205).
