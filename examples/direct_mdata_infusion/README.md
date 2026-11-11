# foxes example: _direct\_mdata\_infusion_

This example reads Weibull states at turbines and directly infuses the data into turbines.

## Check options
Check options by
```
foxes_yaml -h
```

## Run command
```
foxes_yaml inputs.yaml
```

## Automated run

The runner executes this input once, in isolation. Quick mode uses bounded local inputs, no downloads or scheduler jobs. Pass `--nofig` to the runner to suppress display; temporary input and output files are removed on success and failure.

<!-- run-all: foxes_yaml inputs.yaml -->
<!-- run-all-quick: foxes_yaml inputs.yaml -->

The aggregate runner defaults to quiet library execution while announcing each example. Use its `-v N` (or `--verbosity N`) option to request more detail; explicit command verbosity takes precedence.

Levels `2` and `3` print collected farm-results datasets once; levels `0` and
`1` suppress them. The YAML entry point also supports `--verbosity 2` directly.
Writer-only and non-farm selections are not reported as farm datasets.
