# foxes example: _compare gaussian lookup partial wakes_

Compare analytical `gaussian` and `gaussian_lookup` partial-wake models with a
numerical `grid400` rotor reference for the same two-turbine farm and a lateral
scan of the downstream rotor. The analytical cases use the `centre` rotor;
the reference uses `rotor_model="grid400"` and `partial_wakes="rotor_points"`.

The example produces a figure with:

- normalized downstream rotor-effective wind speed from all three models;
- absolute differences of `gaussian` and `gaussian_lookup` from `grid400`.

It also prints the maximum and mean absolute normalized differences. The
Gaussian lookup model uses the bundled model-book lookup artifact.

## Run

Show the comparison plot:

```console
python3 run.py
```

The lateral scan is in rotor-diameter units. Adjust the farm spacing and scan
resolution with `--distance`, `--y-span`, and `--step`.

The `gaussian_lookup` case uses the bundled model-book lookup artifact. Create
custom lookup NetCDF files with the standalone `foxes_create_gaussian_lookup`
tool and pass them to `PartialGaussianLookup` in custom scripts.
