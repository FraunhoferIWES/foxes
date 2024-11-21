# Command line applications

Instead of running *foxes* via a Python script, it can also be run from an input parameter 
file in [YAML](https://de.wikipedia.org/wiki/YAML) format. For this purpose the two
command line applications `foxes_yaml` and `foxes_windio` have been added.

## foxes\_yaml

The command line tool `foxes_yaml` accepts input yaml files that follow a *foxes* specific
structure, that will be described shortly. A file with the name `inputs.yaml` can then be run in a terminal by

```console
foxes_yaml inputs.yaml
```

If the parameter file is located at a different location, the working directory will be set
to the parent directory of the file. For example

```console
foxes_yaml path/to/inputs.yaml
```

will have the working directory `path/to/`, i.e., all _relative_ file paths for reading and writing will be interpreted relative to that directory. However, _absolute_ file paths will not be altered.

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

which will then write all results files to a directory called `results`, relative to the working directory.

The structure of *foxes* yaml files is very close to the *foxes* code base. Let's look at an example:

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
  layouts:          # list functions from foxes.input.farm_layout below
    - function: add_from_file
      file_path: test_farm_67.csv
      turbine_models: [my_turbine]

algorithm:
  algo_type: Downwind
  wake_models: [Bastankhah2014_linear_k004]

calc_farm:
  run: True                     # this triggers algo.calc_farm

outputs:                        # this section is optional
  FarmResultsEval:              # class from foxes.output
    functions:                  # list of functions from that class below
      - name: add_capacity
      - name: add_efficiency
  StateTurbineMap:              # class from foxes.output
    functions:                  # list of functions from that class below
      - name: plot_map          # name of the function
        variable: "EFF"         # specify function parameters here
        cmap: "inferno"
        figsize: [6, 7]
        plt_show: True          # display the created figure
```

Any of the applicable *foxes* classes and functions can be added to the respective section of the input yaml file, together with the specific parameter choices.

Whenever the outputs provided by the `foxes.output` package are sufficient for what you are planning to do, e.g. simple results writing to csv or NetCDF files, `foxes_yaml` might be the easiest way of running *foxes* for you.

## foxes\_windio

The [windio](https://github.com/IEAWindTask37/windIO) framework is an attempt to unify input and output data of software tools in the wind energy community. This framework is also based on yaml files following a specific schema, which is still under development. Currently *foxes* is following a [windIO fork](https://github.com/kilojoules/windIO), which can be installed by

```console
pip install git+https://github.com/kilojoules/windIO@master#egg=windIO
```

_windio_ input can be interpreted and run by foxes via the `foxes_windio` command line tool:

```console
foxes_windio path/to/windio_input.yaml
```

The command line options are very similar to `foxes_yaml`, see above, and

```console
foxes_windio -h
```
