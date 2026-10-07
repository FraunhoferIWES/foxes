# Installation

FOXES supports Python 3.10 through 3.14.

## pip

Install the latest release from [PyPI](https://pypi.org/project/foxes/):

```console
pip install foxes
```

To test the current development branch instead, install it from GitHub:

```console
pip install git+https://github.com/FraunhoferIWES/foxes@dev
```

Wind-farm optimization is provided by
[foxes-opt](https://github.com/FraunhoferIWES/foxes-opt). Install it through the
FOXES extra or directly:

```console
pip install "foxes[opt]"
pip install foxes-opt
```

## conda

Install the latest release from
[conda-forge](https://anaconda.org/conda-forge/foxes):

```console
conda install foxes -c conda-forge
```

For optimization support:

```console
conda install foxes-opt -c conda-forge
```

## Optional dependencies

| Extra | Use |
|---|---|
| `opt` | FOXES Optimization |
| `icon_dream` | ICON-DREAM conversion |
| `era5` | ERA5/metpy support |
| `shp` | Geospatial shapefile support |
| `dask` | Dask engines |
| `mpi` | MPI engine |
| `multiprocess` | Multiprocess engine |
| `ray` | Ray engine |
| `doc` | Documentation tools |
| `dev` | Development tools |
| `test` | Test tools and runtime support |

Install one or more extras with bracket syntax:

```console
pip install "foxes[test,opt]"
```

## Development setup

Contributors must first read
[`AGENTS.md`](https://github.com/FraunhoferIWES/foxes/blob/main/AGENTS.md), the
canonical repository policy. It links the scoped instructions and the
architecture, naming, docstring, and
[development](https://github.com/FraunhoferIWES/foxes/blob/main/docs/development.md)
guides. The development guide defines the `uv` environment and validation
commands.
