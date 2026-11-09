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
