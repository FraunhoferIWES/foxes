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
uv run run.py -nf -o create_turbine_gif
uv run run.py -nf -o farm_results create_turbine_mp4
```

These write `ani_WS.gif` or `ani_WS.mp4`; MP4 requires ffmpeg.
The example prints `Writing <path>` when animation encoding begins.
`-F/--fps` controls playback. `-nf` does not suppress requested animation files.
The old `-a` and `-A` options are removed.
