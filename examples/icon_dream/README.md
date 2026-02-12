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
