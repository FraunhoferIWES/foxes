# Overview - how to setup foxes

Running a wind farm simulation with `foxes` requires the following steps, usually reflected in a Python script, notebook or an input parameter file:

1. _Optional:_ Create the so-called `model book`, which contains all selectable models. You can also choose to rely on pre-defined models and skip this step.
2. Create ambient meteorological conditions, called `states` in `foxes` terminology
3. Create the `wind farm`, collecting all turbine information
4. Create the `algorithm` with its parameters and model choices
5. Run calculations via the algorithm (`algo.calc_farm()`, `algo.calc_points()` or `algo.calc_outputs()`). The outputs can be specified by the parameter `outputs=...` in the calculation functions.

The details will become clear in the sections on [Inputs](inputs.rst), [Outputs](outputs.rst) and [Models](models.rst), and the [Examples](examples.rst).
