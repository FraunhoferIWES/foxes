# Overview - how to setup foxes

Running a wind farm simulation with `foxes` requires the following steps, usually reflected in a Python script, notebook or an input parameter file:

1. _Optional:_ Create the so-called `model book`, which contains all selectable models. You can also choose to rely on pre-defined models and skip this step.
2. Create ambient meteorological conditions, called `states` in `foxes` terminology
3. Create the `wind farm`, collecting all turbine information
4. Create the `algorithm` with its parameters and model choices
5. Run calculations via the algorithm (`algo.calc_farm()`, `algo.calc_points()` or `algo.calc_outputs()`). The outputs can be specified by the parameter `outputs=...` in the calculation functions.

The details will become clear in the sections on [Inputs](inputs.rst), [Outputs](outputs.rst) and [Models](models.rst), and the [Examples](examples.rst).

## Running calculations

The algorithm provides three calculation functions. All three accept the
`outputs` parameter to select output models, either by their name in the model
book or as an output model object. Lists can combine several outputs. See
[Outputs](outputs.rst) for the available choices.

### Farm calculations: `algo.calc_farm()`

The farm calculation evaluates the selected models for all turbines and ambient
states, including wake effects. By default, it returns an `xarray.Dataset` with
physical fields on the dimensions `(state, turbine)`, in the original turbine
order:

```python
farm_results = algo.calc_farm()
```

Use `outputs="farm_results_all"` to retain all available farm fields, for example
when the results will be used in a subsequent point calculation. Pass
`ambient=True` to calculate without wake effects.

### Point calculations: `algo.calc_points()`

The point calculation evaluates the flow field at arbitrary locations, using
previously calculated farm results to determine the wake effects:

```python
farm_results = algo.calc_farm(outputs="farm_results_all")
point_results = algo.calc_points(farm_results, points)
```

Here, `points` is an array of coordinates `(x, y, height)` in metres, with shape
`(n_points, 3)`. State-dependent coordinates with shape
`(n_states, n_points, 3)` are also accepted. By default, the returned
`xarray.Dataset` has physical fields on `(state, point)` and preserves the
supplied point order.

### Output processing: `algo.calc_outputs()`

This function runs selected output models, for example to create tables,
write files or plot results. Existing farm or point results can be supplied
through `farm_results` or `point_results`, respectively. Supplied fields are
reused without repeating their scientific calculation:

```python
mean_farm_results = algo.calc_outputs(
    outputs="farm_results_mean",
    farm_results=farm_results,
)
```

When required results are not supplied, the algorithm calculates the farm fields
and evaluation points requested by the output model. Outputs that need neither
farm fields nor point calculations run without those calculations.

A single output model returning one result returns that value directly.
Multiple output models return a dictionary keyed by result name.
Use `ret_dict=True` with any of the three functions to always retain the result
dictionary.
