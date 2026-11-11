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

Select chunk output models for all three cases and any optional runtime benchmark
with `-o/--outputs` (default: `farm_results`). Writer-only selections run every
case but skip numerical comparisons and their figure. Registered writers reuse
their configured filenames, so successive cases can overwrite earlier files.

Save the comparison figure with `--figure-file comparison.png`, independently
of `-o/--outputs`. Use `farm_results_all` when a report requires extra fields.

## Output selection and automated runs

The default output selection is `farm_results`. Use `-o NAME [NAME ...]` to choose model-book outputs; `-o farm_results_write_n2` is a writer-only example. Reporting that needs unselected fields is skipped with a message. `--nofig` suppresses display without disabling explicitly selected file outputs.

```console
python run.py --nofig
```

The runner executes this example once, in a temporary directory. Its quick mode uses bounded local inputs and a local engine; generated files are removed.

<!-- run-all: python run.py -->
<!-- run-all-quick: python run.py -e numpy -y 0.1 -s 0.1 -->

Verbosity (`-v`/`--verbosity`) defaults to `-v 2` for reports. Levels `2` and `3` print each collected `farm_results` dataset once, even when fields needed by later reports are absent. Levels `0` and `1` do not print datasets; writer-only selections do not invent farm results. Use `0` for quiet execution, `1` for a concise summary, or `3` for library progress. The aggregate runner passes `-v 0` by default, still announces each example, and accepts `-v N` (or `--verbosity N`); explicit example-command choices take precedence.
