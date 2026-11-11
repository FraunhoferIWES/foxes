# foxes example: _timeseries_

Evaluation of a random wind farm for a random time series
 of uniform wind states.

## Check options
Check options by
```
python3 run.py -h
```

## Run command
This runs 100 turbines for 10000 time entries:
```
python3 run.py 100 10000
```

## Output selection and automated runs

The default output selection is `farm_results`. Use `-o NAME [NAME ...]` to choose model-book outputs; `-o farm_results_write_n2` is a writer-only example. Reporting that needs unselected fields is skipped with a message. `--nofig` suppresses display without disabling explicitly selected file outputs.

```console
python run.py 9 100 --nofig
```

The runner executes this example once, in a temporary directory. Its quick mode uses bounded local inputs and a local engine; generated files are removed.

<!-- run-all: python run.py 9 100 -->
<!-- run-all-quick: python run.py 2 3 --seed 0 -e numpy -->

Verbosity (`-v`/`--verbosity`) defaults to `-v 2` for reports. Levels `2` and `3` print each collected `farm_results` dataset once, even when fields needed by later reports are absent. Levels `0` and `1` do not print datasets; writer-only selections do not invent farm results. Use `0` for quiet execution, `1` for a concise summary, or `3` for library progress. The aggregate runner passes `-v 0` by default, still announces each example, and accepts `-v N` (or `--verbosity N`); explicit example-command choices take precedence.
