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

Use `-o/--outputs` with one or more names from `ModelBook.outputs`.
The default is `farm_results`. Collect results while also writing NetCDF files
with `-o farm_results farm_results_write_n1000`.
For file-only output, omit the collector, for example `-o create_gif`.

Writer-only runs skip dataset evaluation and interactive plots. `-nf` does not
disable an explicitly selected writer. Use `create_mp4` for MP4 output
(requires external ffmpeg). The same option is available in all Downwind and
Iterative Python examples; see the [examples documentation](../../docs/source/examples.rst).

Select `create_turbine_gif` or `create_turbine_mp4` to add synchronized
turbine-REWS traces below the flow field:

```console
python3 run.py -nf -o create_turbine_gif
python3 run.py -nf -o farm_results create_turbine_mp4
```

These outputs use `FlowTurbineAnimation` and write `ani_WS.gif` or `ani_WS.mp4`.
They share the flow-only outputs' base filenames, so selecting both variants
of the same format overwrites the earlier output unless filenames are customized.

## Output selection and automated runs

The default output selection is `farm_results`. Use `-o NAME [NAME ...]` to choose model-book outputs; `-o farm_results_write_n2` is a writer-only example. Reporting that needs unselected fields is skipped with a message. `--nofig` suppresses display without disabling explicitly selected file outputs.

```console
python run.py --nofig
```

The runner executes this example once, in a temporary directory. Its quick mode uses bounded local inputs and a local engine; generated files are removed.

<!-- run-all: python run.py -->
<!-- run-all-quick: python run.py -e numpy -->

Verbosity (`-v`/`--verbosity`) defaults to `-v 2` for reports. Levels `2` and `3` print each collected `farm_results` dataset once, even when fields needed by later reports are absent. Levels `0` and `1` do not print datasets; writer-only selections do not invent farm results. Use `0` for quiet execution, `1` for a concise summary, or `3` for library progress. The aggregate runner passes `-v 0` by default, still announces each example, and accepts `-v N` (or `--verbosity N`); explicit example-command choices take precedence.
