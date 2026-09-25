# foxes example: binned field data

Convert the 8000-state timeseries input into spatially weighted wind-speed and
wind-direction bins using `BinnedFieldData`.

## Run command

```text
uv run --no-sync python run.py
```

By default, the example reduces the source states and runs directly from the
resulting `BinnedFieldData` object. Each retained histogram bin becomes a state,
while bin weights remain spatially resolved on the regular support grid. The
source-state reduction is initialized in an engine context, and the support
wind-rose data is prepared there. The Matplotlib canvas is created after the
engine context has closed and then shown with `plt.show()`. The process engine
is supported as well:

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
the file first and then runs from `read_binned_data("binned_data.nc")`. This
factory inspects the artifact's `foxes_state_class` attribute and returns the
matching `BinnedFieldData` or `BinnedPointCloudData` object:

```text
uv run --no-sync python run.py -nf -F --write-nc binned_data.nc --read-nc binned_data.nc
```

The file uses a leading sparse `state` dimension containing the retained flat
histogram-bin indices, followed by the regular-grid dimensions `x`, `y`, and
`height`. Bin centers are not stored; FOXES reconstructs every histogram
variable from its bounds and the sparse state indices. Variables in
`mean_vars` are stored under their original names with dimension `(state,)`.
Each value is the conditional weighted mean over source states and support
points. `weight` contains each spatial bin frequency and is always stored on
the full native support.

Set `write_mean_std=True` to additionally store `<variable>_mean` and
`<variable>_std` on the full support for every output variable. Standard
deviations are weighted population deviations, using circular statistics for
wind direction. These diagnostics are omitted by default. When
`mean_vars=None`, all source output variables not used by `bin_vars` are
included. For an input artifact, `None` selects all stored mean variables not
used as bin axes. Pass an explicit sequence, including an empty sequence, to
restrict that selection. The NetCDF attributes include the compact UTM zone
string, for example `33U`, when the foxes config has a UTM zone set. The
`foxes_state_class` attribute identifies the artifact as `BinnedFieldData` or
`BinnedPointCloudData` and is used by `read_binned_data`.

During initialization, reduced arrays and source support are transferred to
model-scoped loaded data. Before an engine dispatches workers, the canonical
runtime dataset moves to the standard FOXES data stash and the binned states
object releases duplicate in-memory artifact and support references. This
keeps each worker copy of the model lightweight. ``unset_running`` restores
the original references on the controller after execution.

Neither class writes non-finite active-bin statistics. The default
`nan_policy="raise"` stops reduction or artifact loading at the first missing
statistic. Set `nan_policy="interpolate"` to fill missing statistics spatially
with the configured interpolation method and nearest-neighbor fallback.
Regular-grid support cannot remove individual points; use
`BinnedPointCloudData` when the `nan_policy="remove"` behavior is required.
Artifact weights, support coordinates, sparse state indices, and state-only
means must always be valid and are not repaired by a NaN policy.

Bins with zero weight at every support point are omitted independently of the
NaN policy. When spatial diagnostics are enabled, non-finite statistics at
individual zero-weight support points are spatially completed before applying
the policy; only non-finite statistics with non-zero weight count as invalid
data. The same normalization and validation are applied when an existing
artifact is read with `write_mean_std=True`.
