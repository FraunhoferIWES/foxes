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

Select calculation results for all three cases and any optional runtime benchmark
with `-R/--results` (default: `farm_results`). Subsequent `-o/--outputs` defaults
to `comparison`, a downstream-REWS table used by the comparison plot and reports.
Replacing it with a writer still runs every case but omits the comparison.
Registered writers reuse their configured filenames, so successive cases can
overwrite earlier files.

Save the comparison figure with `--figure-file comparison.png`, independently
of `-o/--outputs`. `-nf` suppresses the PNG as well as display.

## Output selection and automated runs

Use `-R/--results NAME [NAME ...]` for the calculation results (default:
`farm_results`). `-r` remains the rotor shortcut. Use `-o/--outputs NAME [NAME ...]` for
subsequent output models (default: `comparison`). Each output receives the collected
`farm_results`, or `None` when that key is absent. Results from both stages are printed
by key, followed by their value on a new line. `-nf/--nofig` skips plots, animations and
their image-only point calculations, including image files. Explicit numerical slices,
point data and NetCDF writers still run.

```console
python run.py --nofig
```

The runner executes this example once, in a temporary directory. Its quick mode uses bounded local inputs and a local engine; generated files are removed.

<!-- run-all: python run.py -->
<!-- run-all-quick: python run.py -e numpy -y 0.1 -s 0.1 -->

Verbosity (`-v`/`--verbosity`) defaults to `2`. Levels `2` and `3` print every returned
result once, including non-farm and aggregated results; levels `0` and `1` suppress
these values. Use `0` for quiet execution, `1` for a concise summary, or `3` for library
progress. The aggregate runner defaults to `-v 0` and preserves explicit verbosity in an
example command.
