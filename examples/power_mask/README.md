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
