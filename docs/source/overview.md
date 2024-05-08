# Overview - how to setup foxes

Running a wind farm simulation with `foxes` requires the following steps, usually reflected in a Python script or notebook:

1. _Optional:_ Create the so-called `model book`, which contains all selectable models. You can also choose to rely on pre-defined models and skip this step.
2. Create ambient meteorological conditions, called `states` in `foxes` terminology
3. Create the `wind farm`, collecting all turbine information
4. Create the `algorithm` with its parameters and model choices
5. Run the farm calculation via the algorithm
6. Optionally evaluate the flow field at arbitrary points of interest, also via the algorithm
7. Post-process the results, for example using the `foxes.output` package

The details will become clear in the subsequent examples within this section.
