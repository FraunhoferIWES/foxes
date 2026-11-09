# foxes example: _dyn_wakes_

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

## Animation outputs

Select the animation output models with `-o/--outputs`:

```console
python3 run.py -nf -o create_turbine_gif
python3 run.py -nf -o create_turbine_mp4
python3 run.py -nf -o farm_results create_turbine_gif create_turbine_mp4
```

The example locally configures `FlowTurbineAnimation` for these names;
`create_gif` and `create_mp4` retain their standard flow-only behavior.
Each animation retains the wind-speed flow field, red rotors, layout labels,
reference marker and synchronized REWS traces for turbines 1, 4 and 7.
Files are `ani_WS.gif` and `ani_WS.mp4` in the configured output directory.
MP4 requires ffmpeg. `-F/--fps` sets the frame rate, defaulting to 4, and
`-R/--ref_xy` sets the reference point. `-nf` suppresses interactive figures,
not selected animation writers. Include `farm_results` to retain result
reporting; omit it for file-only output.
