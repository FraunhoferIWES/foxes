# foxes example: _streamline\_wakes_

This example calculates a 3 x 3 regular wind farm in a heterogeneous time series field,
with wakes that follow the streamlines of the flow.

## Check options
Check options by
```
python3 run.py -h
```

## Run command
For running on 4 cores, use
```
python3 run.py -n 4
```

## Output selection and automated runs

The default output selection is `farm_results`. Use `-o NAME [NAME ...]` to choose model-book outputs; `-o farm_results_write_n2` is a writer-only example. Reporting that needs unselected fields is skipped with a message. `--nofig` suppresses display without disabling explicitly selected file outputs.

```console
python run.py --nofig
```

The runner executes this example once, in a temporary directory. Its quick mode uses bounded local inputs and a local engine; generated files are removed.

<!-- run-all: python run.py -->
<!-- run-all-quick: python run.py -e numpy --n_turbines 2 --wake_length_km 1 -->

Verbosity (`-v`/`--verbosity`) defaults to `-v 2` for reports. Levels `2` and `3` print each collected `farm_results` dataset once, even when fields needed by later reports are absent. Levels `0` and `1` do not print datasets; writer-only selections do not invent farm results. Use `0` for quiet execution, `1` for a concise summary, or `3` for library progress. The aggregate runner passes `-v 0` by default, still announces each example, and accepts `-v N` (or `--verbosity N`); explicit example-command choices take precedence.
