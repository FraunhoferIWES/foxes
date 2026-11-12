# foxes example: _timeseries_

Evaluation of a wind farm for a time series of uniform wind and TI states.

## Check options
Check options by
```
python3 run.py -h
```

## Run command
```
python3 run.py
```

## Select outputs

Use `-R/--results` with one or more calculation output names.
The default is `farm_results`. Collect results while also writing NetCDF files
with `-R farm_results farm_results_write_n1000`.
`-o/--outputs` selects subsequent models, defaulting to `state_turbine_map`.
Use `-o create_mp4` for MP4 output (requires external ffmpeg).
`-nf` skips image writers and their point calculations; numerical writers
remain enabled. See the [examples documentation](../../docs/source/examples.rst).

Select `create_turbine_gif` or `create_turbine_mp4` to add synchronized
turbine-REWS traces below the flow field:

```console
python3 run.py -o create_turbine_gif
python3 run.py -R farm_results -o create_turbine_mp4
```

These outputs use `FlowTurbineAnimation` and write `ani_WS.gif` or `ani_WS.mp4`.
They share the flow-only outputs' base filenames, so selecting both variants
of the same format overwrites the earlier output unless filenames are customized.

## Output selection and automated runs

Use `-R/--results NAME [NAME ...]` for the calculation results (default:
`farm_results`). `-r` remains the rotor shortcut. Use `-o/--outputs NAME [NAME ...]` for
subsequent output models (default: `state_turbine_map`). Each output receives the
collected `farm_results`, or `None` when that key is absent. Results from both stages
are printed by key, followed by their value on a new line. `-nf/--nofig` skips plots,
animations and their image-only point calculations, including image files. Explicit
numerical slices, point data and NetCDF writers still run.

```console
python run.py --nofig
```

The runner executes this example once, in a temporary directory. Its quick mode uses bounded local inputs and a local engine; generated files are removed.

<!-- run-all: python run.py -->
<!-- run-all-quick: python run.py -e numpy -->

Verbosity (`-v`/`--verbosity`) defaults to `2`. Levels `2` and `3` print every returned
result once, including non-farm and aggregated results; levels `0` and `1` suppress
these values. Use `0` for quiet execution, `1` for a concise summary, or `3` for library
progress. The aggregate runner defaults to `-v 0` and preserves explicit verbosity in an
example command.
