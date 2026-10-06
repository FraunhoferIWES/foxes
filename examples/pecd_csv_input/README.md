# foxes example: _pecd\_csv\_input_

This example runs a 3 x 3 wind farm against PECD gridded wind data, prints farm results, evaluates state-resolved flow slices, and displays a mean wind-speed flow plot.

The runner takes one path or packaged states filename for each of the wind-speed and wind-direction CSVs. Its defaults use the reduced sample files packaged under `foxes/data/states`: the first 240 hourly records (10 days) and longitude columns from 3°E through 6°E. Override the defaults with `--wind-speed-file` and `--wind-direction-file`; relative paths use FOXES' configured input directory. Existing local files take precedence; missing files are looked up by basename in the packaged states data. The full PECD CSVs are not included in the FOXES source distribution. By default, both fields are assigned a 100 m height coordinate, regardless of their filenames or recorded measurement heights. Set the `height` argument in `PECDStates` when a different common height coordinate is required. Air density is held constant at 1.225 kg/m3.

To add wind-speed heights calculated with `ABLLogWsProfile`, pass `--extrapolation-heights 80 100 120 --profile-z0 0.05`. The reference layer remains at 100 m, and the requested heights are included in the state grid and flow slices. `--profile-mol` can specify the Monin-Obukhov length; its default selects neutral conditions. Choose `profile-z0` and `profile-mol` for the site and conditions being modeled rather than relying on generic defaults.

## Run command

From this directory, run:

```console
python run.py
```

Use `python run.py --help` to see options for both CSV paths, turbine count, engine, state chunk size, and flow resolutions. The turbine count must be a positive square number. Add `--nofig` to run without displaying the mean flow plot.
