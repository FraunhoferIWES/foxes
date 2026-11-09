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
Animation outputs print `Writing <path>` when encoding begins, independently
of the example's quiet algorithm settings.
`-F/--fps` controls playback, and `-V/--variable` changes the animated field
and filename suffix. Point fields are evaluated during each state step in engine
point chunks; encoding happens after iteration finishes. `--debug` retains wake
point/vector overlays. These scripts support the farm collector and animation
models, not other output writers. The old `-a` and `-A` options are removed.
Both names are configured on this script's local `mbook`, preserving the
original grid, colour limits, quiver arrows, red rotors and turbine labels
without changing global model-book defaults.
