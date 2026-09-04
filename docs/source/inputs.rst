.. _inputs:

Inputs
======

Every *foxes* case needs at least the following two inputs from the user in order
to be able to run: :ref:`Wind farm layouts <inputs:wind-farm-layouts>` and
:ref:`Ambient inflow states <inputs:ambient-inflow-states>`.

Additionally, the applied models might need additional data, for example the power
and thrust curves of the selected turbine types. See the :doc:`Models <models>` section for
additional information on how to provide such inputs.

.. _wind-farm-layouts:

Wind farm layouts
-----------------

The first step is to create an empty wind farm object:

    .. code-block:: python

        farm = foxes.WindFarm()

In *foxes* runs, only one wind farm object is present. However, several
physical wind farms can be added to the object, such that multiple wind farms
are being represented. Turbine types and turbine models can vary for each
wind turbine, such that this is no limitation of usage but merely a *foxes*
code design choice.

Wind turbines are to the wind farm, usually by calling one of the functions
of the sub-package :doc:`foxes.input.farm_layout <_autoapi/foxes/input/farm_layout/index>`. Typical choices are:

* :func:`add_from_csv<foxes.input.farm_layout.add_from_csv>`: Reads a *csv* file, in which each row describes one turbine (also accepts a pandas *DataFrame* instead of the file),
* :func:`add_from_file<foxes.input.farm_layout.add_from_file>`: Similarly, additionally also accepting *json* inputs,
* :func:`add_from_wrf<foxes.input.farm_layout.add_from_wrf>`: Reads a WRF wind farm input folder, optionally with turbine files in TBL format,
* :func:`add_grid<foxes.input.farm_layout.add_grid>`: Adds a regular grid of turbines with identical properties,
* :func:`add_row<foxes.input.farm_layout.add_row>`: Adds a row of turbines with identical properties.
* :func:`add_random<foxes.input.farm_layout.add_random>`: Adds turbines at random positions with identical properties.

A typical example might look like this, see the :doc:`Examples <examples>` page for more examples:

    .. code-block:: python

        foxes.input.farm_layout.add_from_file(
            farm,
            "farm_layout.csv",
            col_x="x",
            col_y="y",
            col_H="H",
            turbine_models=["NREL5MW"],
        )

It is also possible to manually add a single turbine to the wind farm. For doing so,
plug an object of the :class:`Turbine<foxes.core.Turbine>` class into the
:meth:`add_turbine<foxes.core.WindFarm.add_turbine>` function of the
:class:`WindFarm<foxes.core.WindFarm>` class.

Any of the above functions for adding turbines requires a parameter *turbine_models*,
which expects a list of strings that represent the names of the
:ref:`Turbine models <turbine-models>` as appearing in the ModelBook object.

.. _ambient-inflow-states:

Ambient inflow states
---------------------

The atmospheric inflow data are reffered to as *ambient states* or simply as *states*
in *foxes* terminology. They are understood as a list of conditions, which are used
for computing all required background data at any arbitrary evaluation point.

Either those states come with associated statistical weights (for example in the case of
a wind rose), or they do not specify it, in which case they are interpreted as equal weight
conditions (for example in the case of timeseries data).

The full list of currently implemented ambient states can be found in the
:doc:`foxes.input.states <_autoapi/foxes/input/states/index>` sub-package. Typical choices are:

* :class:`Timeseries<foxes.input.states.Timeseries>`: Spatially homogeneous timeseries data,
* :class:`MultiHeightTimeseries<foxes.input.states.MultiHeightTimeseries>`, :class:`MultiHeightNCTimeseries<foxes.input.states.MultiHeightNCTimeseries>`: Height dependent timeseries data,
* :class:`FieldData<foxes.input.states.FieldData>`: Field data, (time, z, y, x) or (time, y, x) dependent.
* :class:`NEWAStates<foxes.input.states.NEWAStates>`: WRF data files in `NEWA <https://map.neweuropeanwindatlas.eu/>`_ format,
* :class:`StatesTable<foxes.input.states.StatesTable>`: Spatially homogeneous data with weights,
* :class:`OnePointFlowTimeseries<foxes.input.states.OnePointFlowTimeseries>`: Horizontally homogeneous data translated into inhomogeneous flow,
* :class:`WeibullSectors<foxes.input.states.WeibullSectors>`: Spatially homogeneous Weibull wind speed distributions organized in wind direction sectors.
* :class:`WRGStates<foxes.input.states.WRGStates>`: Wind resource data, i.e., a regular grid of wind roses expressed via Weibull parameters
