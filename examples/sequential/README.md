# foxes example: _sequential_

An example for a sequential time series evaluation.

## Check options
Check options by
```
python3 run.py -h
```

## Run command
```
python3 run.py
```

Select the original flow animation with named output models:

```console
uv run run.py -o create_gif
uv run run.py --debug -R farm_results -o create_mp4
uv run run.py -o create_gif create_mp4
```

These write `ani_WS.gif` or `ani_WS.mp4`; MP4 requires ffmpeg.
At `-v 2` or higher, animation outputs print `Writing <path>` when encoding
begins, independently of the example's quiet algorithm settings.
`-F/--fps` controls playback, and `-V/--variable` changes the animated field
and filename suffix. Point fields are evaluated during each state step in engine
point chunks; encoding happens after iteration finishes. `--debug` retains wake
point/vector overlays. Farm collectors, tables, maps and writers process the
accumulated iterator data afterward. Other point outputs are unsupported.
`-nf` disables animation outputs before the plugin requests any point calculations.
The old `-a` and `-A` options are removed.
Both names are configured on this script's local `mbook`, preserving the
original grid, colour limits, quiver arrows, red rotors and turbine labels
without changing global model-book defaults.

Animation diagnostics use the script verbosity directly: `-v 1` announces
creation; the default `-v 2` adds writing and successful-completion messages,
frame counts and encoder settings; `-v 3` adds per-frame progress. `-v 0` is quiet.

## Output selection and automated runs

Use `-R/--results NAME [NAME ...]` for the calculation results (default:
`farm_results`). `-r` remains the rotor shortcut. Use `-o/--outputs NAME [NAME ...]` for
subsequent output models (default: `create_turbine_gif`). Each output receives the
collected `farm_results`, or `None` when that key is absent. Results from both stages
are printed by key, followed by their value on a new line. `-nf/--nofig` skips plots,
animations and their image-only point calculations, including image files. Explicit
numerical slices, point data and NetCDF writers still run.

```console
python run.py --nofig
```

The runner executes this example once, in a temporary directory. Its quick mode uses bounded local inputs and a local engine; generated files are removed.

<!-- run-all: python run.py -->
<!-- run-all-quick: python run.py -e numpy -S 3 --start-state 0 -->

Verbosity (`-v`/`--verbosity`) defaults to `2`. Levels `2` and `3` print every returned
result once, including non-farm and aggregated results; levels `0` and `1` suppress
these values. Use `0` for quiet execution, `1` for a concise summary, or `3` for library
progress. The aggregate runner defaults to `-v 0` and preserves explicit verbosity in an
example command.
