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

Use `-R/--results NAME [NAME ...]` for the calculation results (default:
`farm_results`). `-r` remains the rotor shortcut. Use `-o/--outputs NAME [NAME ...]` for
subsequent output models (default: `state_turbine_map`). Each output receives the
collected `farm_results`, or `None` when that key is absent. Results from both stages
are printed by key, followed by their value on a new line. `-nf/--nofig` skips plots,
animations and their image-only point calculations, including image files. Explicit
numerical slices, point data and NetCDF writers still run.

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

Verbosity (`-v`/`--verbosity`) defaults to `2`. Levels `2` and `3` print every returned
result once, including non-farm and aggregated results; levels `0` and `1` suppress
these values. Use `0` for quiet execution, `1` for a concise summary, or `3` for library
progress. The aggregate runner defaults to `-v 0` and preserves explicit verbosity in an
example command.
