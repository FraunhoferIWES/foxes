# foxes example: _gaussian_lookup_partial_wakes_

Run a farm simulation with selectable partial wakes models.

By default this example uses the new `gaussian_lookup` partial wakes model.

## Check options
Check options by
```
uv run python run.py -h
```

## Run command
Default run (uses `gaussian_lookup`):
```
uv run python run.py
```

Compare against axiwake6:
```
uv run python run.py --partial-wakes axiwake6
```

Tune gaussian lookup cutoff for negligible weights:
```
uv run python run.py --partial-wakes gaussian_lookup --min-weight 1e-7
```

Configure the generated lookup table:
```
uv run foxes_create_gaussian_lookup ./custom_gaussian_lookup.nc \
	--min-weight 1e-8 \
	--sigma-over-d-min 0.02 \
	--sigma-over-d-max 20.0 \
	--radial-resolution 0.1 \
	--sigma-resolution 0.05
```

The resolution values are spacings between neighboring normalized lookup
points, not numbers of points. The radial axis uses `R/sigma`; omit
`--r-over-sigma-max` to derive its extent from `min_weight` and the smallest
`sigma/D` value. The sigma axis bounds are set explicitly. Use `--n-rho` to
control radial quadrature accuracy when generating the table; it does not
change the lookup grid size.

Choose out-of-range handling policy explicitly:
```
uv run python run.py --partial-wakes gaussian_lookup --bounds-policy clip
uv run python run.py --partial-wakes gaussian_lookup --bounds-policy raise
```

Load a lookup table created with `foxes_create_gaussian_lookup`:
```
uv run python run.py --lookup-file ./custom_gaussian_lookup.nc --nofig
```

Without `--lookup-file`, the example uses the bundled model-book table.

Note: the model-book lookup is generated on `R/sigma` and `sigma/D` axes. With
`--bounds-policy clip` (default), radial or low-sigma out-of-range points raise
an error if their clipped contribution exceeds `min_weight`. Sigma values above
the generated upper bound use the large-sigma asymptote
`exp(-0.5 * (R/sigma)**2)`.
For a different table, use the standalone `foxes_create_gaussian_lookup` tool with
`--min-weight`, `--sigma-over-d-min`, `--sigma-over-d-max`,
`--radial-resolution`, and `--sigma-resolution`, then provide that artifact to
`PartialGaussianLookup`.
