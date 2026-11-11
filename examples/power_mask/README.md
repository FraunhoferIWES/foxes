# foxes example: _power\_mask_

Demonstrates the effect of using a power mask,
i.e., derating and/or boost.

Boosting ramps smoothly over the final `PowerMask.P_lim` power units below
rated power, reaching the requested `FV.MAX_P` at rated power. Thrust remains
bounded at the Betz optimum even for larger boosts. Requested power is retained;
the resulting thrust is an engineering approximation, not a guarantee of
attainable turbine power. A NaN mask leaves normal operation unchanged.

## Check options
Check options by
```
python3 run.py -h
```

## Run command
```
python3 run.py
```

## Output selection and automated runs

The default output selection is `farm_results`. Use `-o NAME [NAME ...]` to choose model-book outputs; `-o farm_results_write_n2` is a writer-only example. Reporting that needs unselected fields is skipped with a message. `--nofig` suppresses display without disabling explicitly selected file outputs.

Curve comparisons use an explicit no-wake wind-speed scan with default and
power-masked turbines, selecting `TurbineTypeCurves` as the output. Heat maps
select `StateTurbineMap` through `calc_outputs` with supplied farm results;
neither plotting output retains the dataset or launches a hidden simulation.

```console
python run.py --nofig
```

The runner executes this example once, in a temporary directory. Its quick mode uses bounded local inputs and a local engine; generated files are removed.

<!-- run-all: python run.py -->
<!-- run-all-quick: python run.py -e numpy -nt 2 -->

Verbosity (`-v`/`--verbosity`) defaults to `-v 2` for reports. Levels `2` and `3` print each collected `farm_results` dataset once, even when fields needed by later reports are absent. Levels `0` and `1` do not print datasets; writer-only selections do not invent farm results. Use `0` for quiet execution, `1` for a concise summary, or `3` for library progress. The aggregate runner passes `-v 0` by default, still announces each example, and accepts `-v N` (or `--verbosity N`); explicit example-command choices take precedence.
