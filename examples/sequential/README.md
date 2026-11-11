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
uv run run.py -nf -o create_gif
uv run run.py -nf --debug -o farm_results create_mp4
uv run run.py -nf -o create_gif create_mp4
```

These write `ani_WS.gif` or `ani_WS.mp4`; MP4 requires ffmpeg.
At `-v 2` or higher, animation outputs print `Writing <path>` when encoding
begins, independently of the example's quiet algorithm settings.
`-F/--fps` controls playback, and `-V/--variable` changes the animated field
and filename suffix. Point fields are evaluated during each state step in engine
point chunks; encoding happens after iteration finishes. `--debug` retains wake
point/vector overlays. These scripts support the farm collector and animation
models, not other output writers. The old `-a` and `-A` options are removed.
Both names are configured on this script's local `mbook`, preserving the
original grid, colour limits, quiver arrows, red rotors and turbine labels
without changing global model-book defaults.

Animation diagnostics use the script verbosity directly: `-v 1` announces
creation; the default `-v 2` adds writing and successful-completion messages,
frame counts and encoder settings; `-v 3` adds per-frame progress. `-v 0` is quiet.

## Output selection and automated runs

The default output selection is `farm_results create_turbine_gif`. Use `-o NAME [NAME ...]` to choose the farm collector and animation models; `-o create_gif` writes an animation without farm reporting. Reporting that needs unselected fields is skipped with a message. `--nofig` suppresses display without disabling explicitly selected file outputs.

```console
python run.py --nofig
```

The runner executes this example once, in a temporary directory. Its quick mode uses bounded local inputs and a local engine; generated files are removed.

<!-- run-all: python run.py -->
<!-- run-all-quick: python run.py -e numpy -S 3 --start-state 0 -o farm_results -->

Verbosity (`-v`/`--verbosity`) defaults to `-v 2` for reports. Levels `2` and `3` print each collected `farm_results` dataset once, even when fields needed by later reports are absent. Levels `0` and `1` do not print datasets; writer-only selections do not invent farm results. Use `0` for quiet execution, `1` for a concise summary, or `3` for library progress. The aggregate runner passes `-v 0` by default, still announces each example, and accepts `-v N` (or `--verbosity N`); explicit example-command choices take precedence.
