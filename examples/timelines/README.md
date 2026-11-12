# foxes example: _timelines_

Shows dynamic wake development for timeseries states input.

## Check options
Check options by
```
python3 run.py -h
```

## Run command
```
python3 run.py
```

Select the original two-panel flow/REWS animation through model-book outputs:

```console
uv run run.py -o create_turbine_gif
uv run run.py -R farm_results -o create_turbine_mp4
```

These write `ani_WS.gif` or `ani_WS.mp4`; MP4 requires ffmpeg.
`-v 1` announces creation. The default `-v 2` adds `Writing <path>` when encoding
begins, successful completion, frame counts and encoder settings; `-v 3`
adds per-frame composition and encoding progress. `-v 0` is quiet.
`-F/--fps` controls playback and `--ref_xy` sets the reference point.
`-nf` skips animation files and their point calculations.
The old `-a` and `-A` options are removed.

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
<!-- run-all-quick: python run.py -e numpy -nt 9 -S 3 -->

Verbosity (`-v`/`--verbosity`) defaults to `2`. Levels `2` and `3` print every returned
result once, including non-farm and aggregated results; levels `0` and `1` suppress
these values. Use `0` for quiet execution, `1` for a concise summary, or `3` for library
progress. The aggregate runner defaults to `-v 0` and preserves explicit verbosity in an
example command.
