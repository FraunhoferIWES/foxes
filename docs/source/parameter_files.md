# Input parameter files

FOXES can run from [YAML](https://yaml.org/) parameter files through the
`foxes_yaml` and `foxes_windio` command-line tools.

Both entry points print collected farm-results datasets at `--verbosity 2`
and higher, independently of fields needed by later output consumers.
Levels `0` and `1` suppress these dataset reports; writer-only and non-farm
outputs are not reported as farm datasets. The example runner forwards its
selected level unchanged, and the readers reduce scientific component
verbosity internally.

## foxes\_yaml

### Command and options

The command line tool `foxes_yaml` accepts input yaml files that follow a *foxes* specific
structure, that will be described shortly. A file with the name `inputs.yaml` can then be run in a terminal by

```console
foxes_yaml inputs.yaml
```

If the parameter file is located elsewhere, relative input paths are resolved
from its parent directory. For example,

```console
foxes_yaml path/to/inputs.yaml
```

resolves relative input paths from `path/to/`. Absolute paths are unchanged.

The `foxes_yaml` command has multiple options, which can be checked by

```console
foxes_yaml -h
```

For example, it is possible to everrule the `rotor_model` choice of the `inputs.yaml` file by the model choice `grid25`, by

```console
foxes_yaml inputs.yaml -r grid25
```

Also engine choices can be overruled, for example by

```console
foxes_yaml inputs.yaml -e process -n 8
```

for enforcing a parallel run on 8 processors using the `ProcessEngine`.

If you wish to modify the default output directory, you can do so by

```console
foxes_yaml inputs.yaml -o results
```

which sets `results` as the output base directory for relative output paths.

### Input file structure

The structure of *foxes* yaml files is very close to the *foxes* code base. Let's look at an example, available [here](https://github.com/FraunhoferIWES/foxes/blob/main/examples/yaml_input/inputs.yaml) in the *foxes* repository:

```yaml
states:
  states_type: Timeseries               # class from foxes.input.states
  data_source: timeseries_8000.csv.gz   # specify constructor arguments here
  output_vars: [WS, WD, TI, RHO]
  var2col:
    WS: ws
    WD: wd
    TI: ti
  fixed_vars:
    RHO: 1.225

model_book:                 # this section is optional
  turbine_types:            # name of the model book section to be updated
    - name: my_turbine      # name of the new model
      ttype_type: PCtFile   # class from foxes.models.turbine_types
      data_source: NREL-5MW-D126-H90.csv # specify constructor arguments here

wind_farm:
  layouts:    # list functions from foxes.input.farm_layout below
    - function: add_from_file
      file_path: test_farm_67.csv
      turbine_models: [my_turbine]

algorithm:
  algo_type: Downwind
  wake_models: [Bastankhah2014_linear_k004]

calc_farm:    # this section is optional; will run by default
  run: True   # this triggers algo.calc_farm
  outputs: farm_results_analysis  # derived fields trigger TurbineEvaluation

outputs:                          # this section is optional
  - output_type: StateTurbineMap  # class from foxes.models.outputs
    variable: EFF                  # constructor-defined selection
    plot_pars:
      cmap: inferno
      figsize: [6, 7]
    show: True                     # omit or set False to suppress display
```

Any of the applicable *foxes* classes and functions can be added to the respective section of the input yaml file, together with the specific parameter choices.

### Derived turbine fields

Select derived fields through the farm collector. The
`farm_results_analysis` model-book selector requests nominal capacity,
capacity factors, efficiency and weights; output preparation applies
`TurbineEvaluation` automatically. No manual turbine-model registration or
post-calculation evaluation output is needed. Fields keep their state/turbine
dimensions unless aggregation is selected. Farm, cluster and all-turbine
grouping, point aggregation and weighted state means are available through
collector options and model-book presets.
See [Outputs](outputs.rst) for result selection and plotting, and the
[models API](api_models.rst) for the output classes.

### Plot creation and variables

Output results can be stored with `result_labels` under names beginning with
`$` and used in later constructor options, including nested dictionaries.
Array-like results support index expressions such as `$data[0]`.
The YAML-only `output_type: plt` exception forwards a `functions` list directly
to `matplotlib.pyplot`: use `function: figure` with `result_labels: $fig`,
then `object: $fig` and `function: add_subplot` to create shared axes.
Pass those axes into layout/rose constructors, save with the figure's
`savefig` method, and show/close with `plt` functions. These calls default to
outside the engine and are skipped entirely with `--nofig`; stored labels
then resolve to `None`. There is no pyplot output model or registration.
Scientific outputs do not accept imperative method lists.
Matplotlib `savefig` filenames resolve relative to the configured output
directory, including when called on a stored figure.
See the
[combined-plot example](https://github.com/FraunhoferIWES/foxes/blob/main/examples/yaml_input/inputs2.yaml)
for a complete workflow.


## foxes\_windio

FOXES currently reads the schema from the
[EUFLOW windIO fork](https://github.com/EUFLOW/windIO). Install its parser with:

```console
pip install git+https://github.com/EUFLOW/windIO@master#egg=windIO
```

Run a WindIO file with:

```console
foxes_windio path/to/windio_input.yaml
```

Use the example under `examples/windio` as the supported input reference. An
explicit FOXES `wake_averaging` setting takes precedence over inferred WindIO
rotor-averaging choices.

WindIO `model_outputs_specification` entries execute through the ordinary
constructor-defined output lifecycle:

- `turbine_outputs.output_variables` selects `power` and/or
  `rotor_effective_velocity`. `StateTurbineTable` writes only these fields with
  `(turbine, time)` dimensions to `turbine_nc_filename`.
- `flow_field.output_variables` selects `wind_speed` and/or `wind_direction`.
  `SlicesData` writes only these fields with `(z, time, y, x)` dimensions to
  `flow_nc_filename`. The `z` coordinates retain the requested plane order.
- `run_configuration.times_run.subset` supplies unique nonnegative state
  positions when `all_occurences` is false. Both reports retain the subset's
  order; empty, repeated and noninteger positions are rejected.
- `output_folder` selects the output directory; `--output_dir` overrides it.
  `report: false` disables an individual report. Default filenames are
  `turbine_outputs.nc` and `flow_field.nc`.

`--nofig` suppresses figures without disabling these numerical reports.
`--verbosity 0` silences adapter progress and NetCDF-writing messages.

WindIO disables `crop_to_farm` for every `DatasetStates`-derived resource, retaining
horizontal and vertical support beyond the turbine area for flow-field grids.
Explicit data selections and interpolation settings are unchanged. Regular-grid
points outside the resource support still raise an interpolation error instead
of being extrapolated.

Enabled WindIO blockage models automatically select the iterative algorithm and
`ground_mirror` for their induction wakes. This applies to `RankineHalfBody`,
`Rathmann`, `SelfSimilarityDeficit`, and `SelfSimilarityDeficit2020`. Selecting
`None` or `none` leaves blockage disabled. TurbOPark wind deficits also enable
ground mirroring; other per-wake ground settings remain unchanged.

List all command options with:

```console
foxes_windio -h
```
