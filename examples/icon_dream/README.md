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

The default output selection is `farm_results`. Use `-o NAME [NAME ...]` to choose model-book outputs; `-o farm_results_write_n2` is a writer-only example. Reporting that needs unselected fields is skipped with a message. `--nofig` suppresses display without disabling explicitly selected file outputs.

```console
python run.py "ICON_DREAM/*.nc" eww.csv power_curves --nofig
```

The runner executes this example once, in a temporary directory. Its quick mode uses bounded local inputs and a local engine; generated files are removed.

Quick mode generates a synthetic ICON grid, farm table and power curve locally;
it never downloads ICON-DREAM or EuroWindWake data. Normal mode needs the prepared
real files above.

<!-- run-all: python run.py "ICON_DREAM/*.nc" eww.csv power_curves -->
<!-- run-all-quick: python run.py icon.nc farms.csv curves --farms Synthetic -e numpy -->

Verbosity (`-v`/`--verbosity`) defaults to `-v 2` for reports. Levels `2` and `3` print each collected `farm_results` dataset once, even when fields needed by later reports are absent. Levels `0` and `1` do not print datasets; writer-only selections do not invent farm results. Use `0` for quiet execution, `1` for a concise summary, or `3` for library progress. The aggregate runner passes `-v 0` by default, still announces each example, and accepts `-v N` (or `--verbosity N`); explicit example-command choices take precedence.
