# foxes example: _windio_

This example runs an example windio yaml input file.

## Check options
Check options by
```
foxes_windio -h
```

## Run command
```
foxes_windio windio_inputs.yaml
```

The configured reports use constructor-defined `StateTurbineTable` and
`SlicesData` outputs. They write `results/turbine_data.nc` with
`(turbine, time)` fields and `results/flow_field.nc` with `(z, time, y, x)`
fields, respectively. Only the requested output variables and time positions
are retained, in the configured order. `--nofig` keeps both numerical files;
`--verbosity 0` suppresses progress and file-writing messages.

## Automated run

Quick mode inlines the bundled public WindIO includes before calling the
WindIO translator and execution APIs with three time steps. It does not exercise
or require the optional external WindIO YAML loader; normal runs still use it.

The runner executes this input once, in isolation. Quick mode uses bounded local inputs, no downloads or scheduler jobs. Pass `--nofig` to the runner to suppress display; temporary input and output files are removed on success and failure.

<!-- run-all: foxes_windio windio_inputs.yaml -->
<!-- run-all-quick: foxes_windio windio_inputs.yaml -->

The aggregate runner defaults to quiet library execution while announcing each example. Use its `-v N` (or `--verbosity N`) option to request more detail; explicit command verbosity takes precedence.

Levels `2` and `3` print collected farm-results datasets once; levels `0` and
`1` suppress them. The WindIO entry point also supports `--verbosity 2` directly.
Writer-only and non-farm selections are not reported as farm datasets.
