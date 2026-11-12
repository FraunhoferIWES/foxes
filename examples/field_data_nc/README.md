# foxes example: _field\_data\_nc_

This example calculates a 3 x 3 regular wind farm in a heterogeneous time series field, and creates a mean flow figure.

## Check options

Check options by

```console
python3 run.py -h
```

## Run command

For all netCDF file inside the `data` folder to be considered, run without arguments. This runs on a single file:

```console
python3 run.py -f data/data_0.nc
```

## Output selection and automated runs

Use `-R/--results NAME [NAME ...]` for the calculation results (default:
`farm_results`). `-r` remains the rotor shortcut. Use `-o/--outputs NAME [NAME ...]` for
subsequent output models (default: `slices flow_plot_xy_mean`). Each output receives the
collected `farm_results`, or `None` when that key is absent. Results from both stages
are printed by key, followed by their value on a new line. `-nf/--nofig` skips plots,
animations and their image-only point calculations, including image files. Explicit
numerical slices, point data and NetCDF writers still run.

```console
python run.py --nofig
```

The runner executes this example once, in a temporary directory. Its quick mode uses bounded local inputs and a local engine; generated files are removed.

<!-- run-all: python run.py -->
<!-- run-all-quick: python run.py -e numpy -nt 1 -f field.nc -->

Verbosity (`-v`/`--verbosity`) defaults to `2`. Levels `2` and `3` print every returned
result once, including non-farm and aggregated results; levels `0` and `1` suppress
these values. Use `0` for quiet execution, `1` for a concise summary, or `3` for library
progress. The aggregate runner defaults to `-v 0` and preserves explicit verbosity in an
example command.
