# foxes example: _sequential\_yawcontroller_

An example for a basic yaw controller.

## Check options
Check options by
```
python3 run.py -h
```

## Run command
```
python3 run.py
```

Select the original flow animation through model-book outputs:

```console
uv run run.py -nf -s my_states.csv -o create_gif
uv run run.py -nf -s my_states.csv --debug -o farm_results create_mp4
uv run run.py -nf -s my_states.csv -o create_gif create_mp4
```

Supply a time-indexed CSV with `WS` and `WD` columns.
These write `ani_WS.gif` or `ani_WS.mp4`; MP4 requires ffmpeg.
Animation outputs print `Writing <path>` when encoding begins, independently
of the example's quiet algorithm settings.
`-F/--fps` controls playback and `--debug` retains wake point/vector overlays.
Point fields are evaluated during state-wise iteration with engine point
chunking, then encoded after completion. Output selection supports the farm
collector and animation models, not other writers. The old `-a` and `-A`
options are removed.
Both names are configured on the local `mbook`, preserving the original flow
grid, quiver arrows, red rotors and turbine labels without changing global
model-book defaults.
