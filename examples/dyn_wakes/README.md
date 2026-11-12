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
python3 run.py -o create_turbine_gif
python3 run.py -o create_turbine_mp4
python3 run.py -R farm_results -o create_turbine_gif create_turbine_mp4
```

The example locally configures `FlowTurbineAnimation` for these names;
`create_gif` and `create_mp4` retain their standard flow-only behavior.
Each animation retains the wind-speed flow field, red rotors, layout labels,
reference marker and synchronized REWS traces for turbines 1, 4 and 7.
Files are `ani_WS.gif` and `ani_WS.mp4` in the configured output directory.
MP4 requires ffmpeg. `-F/--fps` sets the frame rate, defaulting to 4, and
`--ref_xy` sets the reference point. `-R/--results` selects calculation results
independently. `-nf` skips animations and their point calculations entirely.

`-v 1` announces animation creation.
The default `-v 2` adds `Writing <path>`, successful completion, frame counts,
encoder and frame rate; `-v 3` adds per-frame composition and encoding progress.
`-v 0` is quiet.

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
