# foxes example: _icon\_dream_

This example computes an offshore wind farm with inflow data from
the ICON-DREAM dataset.

## Preparations

1. Download ICON-DREAM data using the provided tool `iconDream2foxes`, e.g. for
a time period January till March 2024:

```console
iconDream2foxes "/path/to/icon/dream/output/folder" baltic 2024 1 2024 3
```

2. Download the EuroWindWake wind farm data base, using the provided tool `eww2foxes`:

```console
eww2foxes "/path/to/eww/output/folder"
```

## Check options

Check options by

```console
python3 run.py -h
```

## Run the calculation

Now run the `run.py` script, giving it the above folders, e.g.

```console
"/path/to/icon/dream/output/folder/nc/baltic" "/path/to/eww/output/folder/eww_opendatabase.csv" "/path/to/eww/output/folder/power_curves"
```

## Output selection and automated runs

Use `-R/--results NAME [NAME ...]` for the calculation results (default:
`farm_results`). `-r` remains the rotor shortcut. Use `-o/--outputs NAME [NAME ...]` for
subsequent output models (default: `state_turbine_table`). Each output receives the
collected `farm_results`, or `None` when that key is absent. Results from both stages
are printed by key, followed by their value on a new line. `-nf/--nofig` skips plots,
animations and their image-only point calculations, including image files. Explicit
numerical slices, point data and NetCDF writers still run.

```console
python run.py "ICON_DREAM/*.nc" eww.csv power_curves --nofig
```

The runner executes this example once, in a temporary directory. Its quick mode uses bounded local inputs and a local engine; generated files are removed.

Quick mode generates a synthetic ICON grid, farm table and power curve locally;
it never downloads ICON-DREAM or EuroWindWake data. Normal mode needs the prepared
real files above.

<!-- run-all: python run.py "ICON_DREAM/*.nc" eww.csv power_curves -->
<!-- run-all-quick: python run.py icon.nc farms.csv curves --farms Synthetic -e numpy -->

Verbosity (`-v`/`--verbosity`) defaults to `2`. Levels `2` and `3` print every returned
result once, including non-farm and aggregated results; levels `0` and `1` suppress
these values. Use `0` for quiet execution, `1` for a concise summary, or `3` for library
progress. The aggregate runner defaults to `-v 0` and preserves explicit verbosity in an
example command.
