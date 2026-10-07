# Input parameter files

FOXES can run from [YAML](https://yaml.org/) parameter files through the
`foxes_yaml` and `foxes_windio` command-line tools.

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

outputs:                          # this section is optional
  - output_type: FarmResultsEval  # class from foxes.output
    functions:                    # list of functions from that class below
      - function: add_capacity
      - function: add_efficiency
  - output_type: StateTurbineMap  # class from foxes.output
    functions:                    # list of functions from that class below
      - function: plot_map        # name of the function
        variable: "EFF"           # specify function parameters here
        cmap: "inferno"
        figsize: [6, 7]
  - output_type: plt              # class from foxes.output
    functions:                    # list of functions from that class below
      - function: show            # runs plt.show(), triggering the display
      - function: close           # runs plt.close(), optional
```

Any of the applicable *foxes* classes and functions can be added to the respective section of the input yaml file, together with the specific parameter choices.

### Output evaluation

The `outputs` section can call FOXES output classes, including
`FarmResultsEval` for reductions, efficiency, capacity, and yield. See the
[output API](api_output.rst) for available methods and arguments. Calculations
that need Python objects such as the algorithm are usually clearer in a script
or notebook.

### Plot creation and variables

Function results can be stored under names beginning with `$` and passed to
later calls. Use `object: $name` to call methods on a stored object; array-like
results also support index expressions such as `$data[0]`. See the
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
rotor-averaging choices. List all command options with:

```console
foxes_windio -h
```
