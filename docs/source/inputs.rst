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
plug an object of the :class:`Turbine<foxes.core.turbine.Turbine>` class into the
:meth:`add_turbine<foxes.core.WindFarm.add_turbine>` function of the
:class:`WindFarm<foxes.core.wind_farm.WindFarm>` class.

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

* :class:`Timeseries<foxes.input.states.states_table.Timeseries>`: Spatially homogeneous timeseries data,
* :class:`MultiHeightTimeseries<foxes.input.states.multi_height.MultiHeightTimeseries>`, :class:`MultiHeightNCTimeseries<foxes.input.states.multi_height.MultiHeightNCTimeseries>`: Height dependent timeseries data,
* :class:`FieldData<foxes.input.states.field_data.FieldData>`: Field data, (time, z, y, x) or (time, y, x) dependent.
* :class:`NEWAStates<foxes.input.states.newa_states.NEWAStates>`: WRF data files in `NEWA <https://map.neweuropeanwindatlas.eu/>`_ format,
* :class:`StatesTable<foxes.input.states.states_table.StatesTable>`: Spatially homogeneous data with weights,
* :class:`OnePointFlowTimeseries<foxes.input.states.one_point_flow.OnePointFlowTimeseries>`: Horizontally homogeneous data translated into inhomogeneous flow,
* :class:`WeibullSectors<foxes.input.states.weibull_sectors.WeibullSectors>`: Spatially homogeneous Weibull wind speed distributions organized in wind direction sectors.
* :class:`WRGStates<foxes.input.states.wrg_states.WRGStates>`: Wind resource data, i.e., a regular grid of wind roses expressed via Weibull parameters

Creating a single mean field
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`create_dataset_mean_from_states<foxes.input.states.create.create_dataset_mean_from_states>`
function evaluates any states model on a Cartesian grid and reduces its state
dimension using the state weights. Wind speed and direction are averaged as
vectors and written as ``WS`` and ``WD``. By default, ``MAIN_WD`` additionally
contains the circular mean of the direction sector with the greatest total
state weight. Sectors overlap by 50 percent so a mode crossing one sector
boundary is centred in another sector. Their minimum width is controlled by
``wd_histo_width``; set ``vname_main_wd=None`` to disable the calculation. The
grid is read from ``micro_states``. Complete Cartesian support
retains its native axes; irregular support is converted to a regular grid using
:func:`regular_grid_from_points<foxes.utils.regular_grid_from_points>`. An
optional area geometry crops the horizontal axes while retaining one exterior
grid point on every side. Point and state batches bound the temporary evaluation
data, and the result can be used directly by
:class:`SingleStateField<foxes.input.states.SingleStateField>`:

    .. code-block:: python

        import numpy as np
        import pandas as pd
        import xarray as xr

        import foxes
        import foxes.variables as FV

        states = foxes.input.states.StatesTable(
            data_source=pd.DataFrame(
                {
                    FV.WS: [8.0, 10.0],
                    FV.WD: [260.0, 280.0],
                    FV.WEIGHT: [0.4, 0.6],
                }
            ),
            output_vars=[FV.WS, FV.WD],
        )
        grid = xr.Dataset(
            data_vars={
                FV.WS: (
                    ("state", "height", "y", "x"),
                    np.full((1, 1, 11, 11), 8.0),
                ),
                FV.WD: (
                    ("state", "height", "y", "x"),
                    np.full((1, 1, 11, 11), 270.0),
                ),
            },
            coords={
                "state": [0],
                "x": np.arange(0.0, 1001.0, 100.0),
                "y": np.arange(0.0, 1001.0, 100.0),
                "height": [100.0],
            },
        )
        micro_states = foxes.input.states.FieldData(
            data_source=grid,
            output_vars=[FV.WS, FV.WD],
            states_coord="state",
            x_coord="x",
            y_coord="y",
            h_coord="height",
            time_format=None,
        )
        with foxes.Engine.new("default", verbosity=0):
            mean_data = foxes.input.states.create.create_dataset_mean_from_states(
                states=states,
                micro_states=micro_states,
                output_vars=[FV.WS, FV.WD],
                wd_histo_width=30.0,
                verbosity=0,
            )

        mean_states = foxes.input.states.SingleStateField(
            data_source=mean_data,
            output_vars=[FV.WS, FV.WD],
        )

Use :func:`detect_regular_grid<foxes.utils.detect_regular_grid>` to inspect
explicit three-dimensional support points without selecting a replacement grid,
or :func:`select_grid_axes<foxes.utils.select_grid_axes>` to crop known native
axes directly. The ``grid_point_plot`` argument of
``create_dataset_mean_from_states`` writes a proof plot containing source
support points, selected mean-field points, and the boundary when supplied.
