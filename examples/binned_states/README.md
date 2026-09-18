# foxes example: _binned_states_

Convert the 8000-state timeseries input into weighted wind-speed and wind-direction bins using `BinnedStates`.

## Run command

```text
uv run --no-sync python run.py
```

By default, the example reduces the source states and runs directly from the
resulting `BinnedStates` object. The source-state reduction is initialized in an
engine context, and the support wind-rose data is prepared there. The Matplotlib
canvas is created after the engine context has closed and then shown with
`plt.show()`. The process engine is supported as well:

```text
uv run --no-sync python run.py -e process
```

Run the binned-data calculation and compare weighted means against the full
timeseries calculation with:

```text
uv run --no-sync python run.py -nf -F
```

Use `--write-nc binned_data.nc` to write the reduced binned data to a NetCDF
artifact. Use `--read-nc binned_data.nc` to run from a previously written
artifact. The options are independent and default to `None`; using both writes
the file first and then runs from `BinnedStates("binned_data.nc")`:

```text
uv run --no-sync python run.py -nf -F --write-nc binned_data.nc --read-nc binned_data.nc
```

The file uses bin dimensions followed by either the regular-grid or
scattered-point support topology, with `*_min`, `*_mean`, `*_max`, and `weight`
data variables. The NetCDF attributes include the compact UTM zone string, for
example `33U`, when the foxes config has a UTM zone set.
