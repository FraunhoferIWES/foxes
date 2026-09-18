# foxes example: _binned_states_

Convert the 8000-state timeseries input into weighted wind-speed and wind-direction bins using `BinnedStates`.

## Run command

```text
uv run --no-sync python run.py
```

The source-state reduction is initialized in an engine context, the support
wind-rose data is prepared there. The Matplotlib canvas is created after the
engine context has closed and then shown with `plt.show()`. The process engine
is supported as well:

```text
uv run --no-sync python run.py -e process
```

Set `BinnedStates(..., output_file="binned_states.nc")` to persist the reduced
support-point data as soon as it is available. Regular support grids are written
in a `FieldData`-readable layout; scattered support points are written in a
`PointCloudData`-readable layout. The NetCDF attributes include the compact UTM
zone string, for example `33U`, when the foxes config has a UTM zone set.
