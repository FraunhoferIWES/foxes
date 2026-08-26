# foxes example: _compare gaussian lookup partial wakes_

Compare the `gaussian_lookup` and `axiwake6` partial-wake models for the same
two-turbine farm and a lateral scan of the downstream rotor.

The example produces a figure with:

- normalized downstream rotor-effective wind speed from both models;
- the absolute difference between the two results.

It also prints the maximum and mean absolute normalized difference. The
Gaussian lookup model uses the bundled model-book lookup artifact.

## Run

Show the comparison plot:

```console
uv run python run.py
```

The lateral scan is in rotor-diameter units. Adjust the farm spacing and scan
resolution with `--distance`, `--y-span`, and `--step`.

The `gaussian_lookup` case uses the bundled model-book lookup artifact. Create
custom lookup NetCDF files with the standalone `foxes_create_gaussian_lookup`
tool and pass them to `PartialGaussianLookup` in custom scripts.
