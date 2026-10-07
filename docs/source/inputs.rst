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

State support-point plots
^^^^^^^^^^^^^^^^^^^^^^^^^

``FieldData``, ``LatLonFieldData``, ``PointCloudData``, ``WeibullPointCloud``,
``BinnedFieldData``, ``BinnedPointCloudData``, and ``TurbinePointCloud`` accept
``grid_point_plot`` to write a support-point image with the farm layout.
Point-cloud plots project selected support heights onto the horizontal plane;
ordinary and binned clouds use the first loaded dataset, whereas turbine-backed
clouds plot the current farm turbine locations during loading. The latter does
not represent subsequent state-dependent or optimization layouts. ``None``
disables these diagnostic plots. ``NEWAStates`` uses ``wrf_point_plot``, and
``MesoMicroField`` uses ``support_point_plot`` for micro support and reference
points together.

The corresponding ``grid_point_plot_pars``, ``wrf_point_plot_pars``, and
``support_point_plot_pars`` control support markers. Farm overlays have separate
``grid_point_plot_farm_pars``, ``wrf_point_plot_farm_pars``, and
``support_point_plot_farm_pars`` dictionaries passed to
:meth:`FarmLayoutOutput.get_figure<foxes.output.FarmLayoutOutput.get_figure>`.
For an exceptional title-free plot with invisible turbines, pass
``{"title": "", "alpha": 0.0, "annotate": 0}`` as the farm-overlay dictionary.
This keeps the support/reference markers visible. Defaults retain the usual
farm title, visible turbines, and annotation behavior. Parameter dictionaries
are copied, and non-dictionary values raise ``TypeError``. Direct parameters
passed to ``MesoMicroField.get_support_point_figure`` override its configured
farm-overlay defaults.

Dense CFD grids can otherwise merge into a solid colour at image resolution.
``FieldData.grid_point_plot_stride`` and
``MesoMicroField.support_point_plot_stride`` accept a positive integer sampling
step along each horizontal axis. The default ``1`` draws every selected point;
for example, ``15`` draws every fifteenth coordinate and retains the outermost
coordinates. Sampling applies only to the diagnostic image, never to loaded CFD
data, interpolation, or reference points. Combine it with small, edge-free
markers such as ``{"markersize": 2.0, "markeredgewidth": 0.0}``.

To draw an outline above the points without hiding them, set the overlay's
``bargs`` to ``{"show_boundary": True, "fill_mode": None}`` and use
``pars_boundary`` with ``facecolor="none"`` and a higher ``zorder`` than the
markers. The optional ``bargs["boundary"]`` overrides the boundary for the
figure only, including plots made with a temporary loading farm that has no
boundary. Passing ``None`` disables that figure's boundary; omitting the key
uses the farm boundary. Numerical farm bounds are never changed by this option.

Mean-flow plot coverage
^^^^^^^^^^^^^^^^^^^^^^

For mean-flow plots extending beyond farm bounds, ``bounds_extra_space=None``
on the micro ``FieldData`` retains its full native horizontal grid. Extrapolation
beyond the selected CFD grid can produce extreme values that dominate the plot's
colour scale. To leave unsupported regions blank instead, explicitly pass
``interp_pars={"bounds_error": False, "fill_value": np.nan}``, with NumPy
imported as ``np``. ``MesoMicroField`` reference points must remain inside the
selected micro grid. These options change data selection and interpolation,
not just figure styling; they are opt-in and do not change normal defaults.

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
