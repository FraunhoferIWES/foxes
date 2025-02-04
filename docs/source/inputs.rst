Inputs
======

Every *foxes* case needs at least the following two inputs from the user in order 
to be able to run: :ref:`Wind farm layouts` and :ref:`Ambient inflow states`. 

Additionally, the applied models might need additional data, for example the power 
and thrust curves of the selected turbine types. See the :ref:`Models` Section for 
additional information on how to provide such inputs.

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
of the sub package :ref:`foxes.input.farm_layout`. Typical choices are:

* :ref:`add_from_csv<foxes.input.farm_layout.add_from_csv>`: Reads a *csv* file, in which each row describes one turbine (also accepts a pandas *DataFrame* instead of the file),
* :ref:`add_from_file<foxes.input.farm_layout.add_from_file>`: Similarly, additionally also accepting *json* inputs,
* :ref:`add_grid<foxes.input.farm_layout.add_grid>`: Adds a regular grid of turbines with identical properties,
* :ref:`add_row<foxes.input.farm_layout.add_row>`: Adds a row of turbines with identical properties.
* :ref:`add_random<foxes.input.farm_layout.add_random>`: Adds turbines at random positions with identical properties.

A typical example might look like this, see :ref:`Examples` for more examples:

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
plug an object of the :ref:`Turbine<foxes.core.Turbine>` class into the
:ref:`add_turbine<foxes.core.WindFarm.add_turbine>` function of the 
:ref:`WindFarm<foxes.core.WindFarm>` class.

Any of the above functions for adding turbines requires a parameter *turbine_models*,
which expects a list of strings that represent the names of the :ref:`Turbine models`
as appearing in the :ref:`ModelBook object<The model book>`.

Ambient inflow states
---------------------

The atmospheric inflow data are reffered to as *ambient states* or simply as *states* 
in *foxes* terminology. They are understood as a list of conditions, which are used
for computing all required background data at any arbitrary evaluation point. 

Either those states come with associated statistical weights (for example in the case of
a wind rose), or they do not specify it, in which case they are interpreted as equal weight 
conditions (for example in the case of timeseries data).

The full list of currently implemented ambient states can be found in the
:ref:`foxes.input.states` sub package. Typical choices are:

* :ref:`Timeseries<foxes.input.states.Timeseries>`: Spatially homogeneous timeseries data, see :ref:`Timeseries data`,
* :ref:`MultiHeightTimeseries<foxes.input.states.MultiHeightTimeseries>`, :ref:`MultiHeightNCTimeseries<foxes.input.states.MultiHeightNCTimeseries>`: Height dependent timeseries data, see :ref:`Multi-height wind data`,
* :ref:`FieldDataNC<foxes.input.states.FieldDataNC>`: Field data, (time, z, y, x) or (time, y, x) dependent. See :ref:`Heterogeneous flow`,
* :ref:`StatesTable<foxes.input.states.StatesTable>`: Spatially homogeneous data with weights, see :ref:`Wind rose data`,
* :ref:`OnePointFlowTimeseries<foxes.input.states.OnePointFlowTimeseries>`: Horizontally homogeneous data translated into inhomogeneous flow, see :ref:`Dynamic Wakes 1`,
* :ref:`WRGStates<foxes.input.states.WRGStates>`: Wind resource data, i.e., a regular grid of wind roses.
