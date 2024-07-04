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

## Requirements

The supported Python versions are: 

- `Python 3.8`
- `Python 3.9`
- `Python 3.10`
- `Python 3.11`
- `Python 3.12`

## Installation

Either install via pip:

```console
pip install foxes
```

Alternatively, install via conda:

```console
conda install foxes -c conda-forge
```

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
