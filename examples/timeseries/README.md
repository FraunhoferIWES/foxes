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
