# Welcome to FOXES

![FOXES Logo](Logo_FOXES.svg)

## Overview

*Farm Optimization and eXtended yield Evaluation Software*

*FOXES* is a modular Python package for wind-farm and wake modelling by
Fraunhofer IWES. Typical applications include:

- Wind farm optimization, e.g. layout optimization or wake steering,
- Wind farm post-construction analysis,
- Wake model studies, comparison and validation,
- Wind farm simulations invoking complex model chains.

FOXES uses vectorized calculations and supports local or distributed execution.
Wind-farm optimization is provided by the separate
[foxes-opt](https://github.com/FraunhoferIWES/foxes-opt) package.

| Resource | Link |
| :--- | :--- |
| Documentation | [https://fraunhoferiwes.github.io/foxes](https://fraunhoferiwes.github.io/foxes) |
| Source code | [https://github.com/FraunhoferIWES/foxes](https://github.com/FraunhoferIWES/foxes) |
| PyPI | [https://pypi.org/project/foxes/](https://pypi.org/project/foxes/) |
| conda-forge | [https://anaconda.org/conda-forge/foxes](https://anaconda.org/conda-forge/foxes) |
| License | [MIT](https://github.com/FraunhoferIWES/foxes/blob/main/LICENSE) |

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/FraunhoferIWES/foxes)

## Citation

Please cite the JOSS paper "FOXES: Farm Optimization and eXtended yield
Evaluation Software".

[![DOI](https://joss.theoj.org/papers/10.21105/joss.05464/status.svg)](https://doi.org/10.21105/joss.05464)

BibTeX:
```bibtex
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

Requires Python 3.10 through 3.14.

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

For detailed examples of how to run _foxes_, check the [examples](examples/)
and [notebooks](notebooks/) folders in this repository. A minimal running
example is the following, based on provided static `csv` data files:

```python
import foxes

if __name__ == "__main__":
    states = foxes.input.states.Timeseries(
        "timeseries_3000.csv.gz", ["WS", "WD", "TI", "RHO"]
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_from_file(
        farm, "test_farm_67.csv", turbine_models=["NREL5MW"]
    )

    algo = foxes.algorithms.Downwind(farm, states, ["Jensen_linear_k007"])
    farm_results = algo.calc_farm()

    print(farm_results)
```

## Testing

For testing, please clone the repository and install the required dependencies:

```console
git clone https://github.com/FraunhoferIWES/foxes.git
cd foxes
uv sync --extra test
```

Run the tests with:

```console
uv run pytest tests
```

For the full development setup, see the [development guide](docs/development.md).

## Contributing

1. Fork _foxes_ on _github_.
2. Create a branch (`git checkout -b new_branch`)
3. Commit your changes (`git commit -am "your awesome message"`)
4. Push to the branch (`git push origin new_branch`)
5. Create a pull request [here](https://github.com/FraunhoferIWES/foxes/pulls)

## Acknowledgements

Development of *FOXES* was supported by:

- BMWK projects *Smart Wind Farms* (0325851B), *GW-Wakes* (0325397B), and *X-Wakes* (03EE3008A)
- BMBF project *H2Digital* (03SF0635)
- Horizon Europe projects *FLOW* (101084205) and *AIRE* (101083716)
