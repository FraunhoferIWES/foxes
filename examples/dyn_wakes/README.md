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

`-v 1` announces animation creation.
The default `-v 2` adds `Writing <path>`, successful completion, frame counts,
encoder and frame rate; `-v 3` adds per-frame composition and encoding progress.
`-v 0` is quiet.

## Output selection and automated runs

The default output selection is `farm_results`; select animation outputs explicitly. Use `-o NAME [NAME ...]` to choose model-book outputs; `-o farm_results_write_n2` is a writer-only example. Reporting that needs unselected fields is skipped with a message. `--nofig` suppresses display without disabling explicitly selected file outputs.

```console
python run.py --nofig
```

The runner executes this example once, in a temporary directory. Its quick mode uses bounded local inputs and a local engine; generated files are removed.

<!-- run-all: python run.py -->
<!-- run-all-quick: python run.py -e numpy -nt 4 -S 3 -o farm_results -->

Verbosity (`-v`/`--verbosity`) defaults to `-v 2` for reports. Levels `2` and `3` print each collected `farm_results` dataset once, even when fields needed by later reports are absent. Levels `0` and `1` do not print datasets; writer-only selections do not invent farm results. Use `0` for quiet execution, `1` for a concise summary, or `3` for library progress. The aggregate runner passes `-v 0` by default, still announces each example, and accepts `-v N` (or `--verbosity N`); explicit example-command choices take precedence.
