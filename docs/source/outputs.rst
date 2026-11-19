.. _outputs:

Outputs
=======

The outputs of *foxes* runs are selected by the user through the ``outputs``
parameter of ``algo.calc_farm()``, ``algo.calc_points()`` or
``algo.calc_outputs()``. They can include result datasets, files, tables and
figures. See the :doc:`Overview <overview>` for the calculation functions.

All concrete output models are implemented in the
:doc:`foxes.models.outputs <_autoapi/foxes/models/outputs/index>` sub-package.
See also the :download:`outputs example notebook <notebooks/outputs.ipynb>`.

.. _output-selection:

Selecting outputs
-----------------

Output models are stored in the :class:`ModelBook <foxes.models.model_book.ModelBook>`
object under a name string, in the same way as the :doc:`Models <models>`.
The ``outputs`` parameter accepts a model-book output name, an output model
object, or a list of names and objects.

For example, the default ``farm_results`` output returns farm results as an
``xarray.Dataset``:

    .. code-block:: python

        farm_results = algo.calc_farm(outputs="farm_results")

To configure an output directly, pass an object from the output sub-package.
For example, the following selects turbine power with and without wake effects:

    .. code-block:: python

        import foxes.variables as FV
        from foxes.models.outputs import FarmResultsOutput

        power_results = algo.calc_farm(
            outputs=FarmResultsOutput(variables=[FV.P, FV.AMB_P]),
        )

Note that ``outputs`` selects output models, not physical variable names.
Variable selections belong to the constructor of the selected output model.

A single output model returning one result returns that value directly.
Multiple output models return a dictionary whose keys are result names, not
necessarily the selected model-book names. Use ``ret_dict=True`` to retain
this dictionary also for a single output. Writer-only outputs return an empty
dictionary. Setting ``outputs=None`` or ``outputs=[]`` disables collection.

.. _model-book-outputs:

Model-book outputs
------------------

Typical model-book choices are shown below, with at least one entry for every
concrete output class. ``Has points`` indicates that the selected output uses
flow data at evaluation points, rather than only turbine data or model metadata.

.. list-table::
    :header-rows: 1
    :widths: 30 60 10

    * - Name
      - Description
      - Has points
    * - ``farm_results``
      - Collect the default selection of farm variables.
      - ``False``
    * - ``farm_results_all``
      - Collect all available farm variables.
      - ``False``
    * - ``farm_results_mean``
      - Compute weighted state means for each turbine.
      - ``False``
    * - ``farms_results``
      - Aggregate turbines by their wind-farm membership.
      - ``False``
    * - ``cluster_results``
      - Aggregate turbines by their cluster membership.
      - ``False``
    * - ``total_results``
      - Aggregate all turbines.
      - ``False``
    * - ``farm_results_write_n1000``
      - Write farm results to NetCDF files with up to 1,000 states per file.
      - ``False``
    * - ``farm_results_write_chunks``
      - Write one farm-result NetCDF file per engine state chunk.
      - ``False``
    * - ``farm_results_write_month``
      - Write farm results to monthly NetCDF files.
      - ``False``
    * - ``point_results``
      - Collect the default selection of point variables.
      - ``True``
    * - ``point_results_write_n1000``
      - Write point results to NetCDF files with up to 1,000 states per file.
      - ``True``
    * - ``point_results_write_chunks``
      - Write one point-result NetCDF file per engine state/point chunk.
      - ``True``
    * - ``point_results_write_month``
      - Write point results to monthly NetCDF files.
      - ``True``
    * - ``farm_layout``
      - Plot the wind farm layout.
      - ``False``
    * - ``rose_plot``
      - Plot a wind rose from turbine data.
      - ``False``
    * - ``support_rose_plot``
      - Plot native wind roses at the support points of binned states.
      - ``False``
    * - ``states_rose_plot``
      - Evaluate states at one configured location and plot a wind rose.
      - ``True``
    * - ``wind_rose_bin_plot``
      - Plot a turbine variable reduced within wind direction and speed bins.
      - ``False``
    * - ``state_turbine_map``
      - Plot a variable against state and turbine indices.
      - ``False``
    * - ``state_turbine_table``
      - Create a table of state and turbine data.
      - ``False``
    * - ``point_calculator``
      - Collect configured or supplied point data, optionally as state means.
      - ``True``
    * - ``rotor_point_plot``
      - Plot rotor quadrature points and weights without evaluating the flow.
      - ``False``
    * - ``turbine_type_curves``
      - Plot turbine-type performance curves.
      - ``False``
    * - ``slice_data_xy_states``
      - Collect state-resolved flow data on a horizontal plane.
      - ``True``
    * - ``slices_data_xy_mean``
      - Collect state-mean flow data on several horizontal planes.
      - ``True``
    * - ``flow_plot_xy_mean``
      - Plot state-mean flow data on a horizontal plane.
      - ``True``
    * - ``create_gif``
      - Write a planar flow animation as a GIF.
      - ``True``
    * - ``create_turbine_gif``
      - Write a GIF combining a flow animation with turbine-variable traces.
      - ``True``

For outputs marked ``True``, prefer ``algo.calc_outputs()`` over
``algo.calc_farm()`` when calculating their point data. This lets the engine
chunk in both the state and point dimensions, instead of evaluating point
blocks inside each farm-state chunk. Independent point chunks can improve
parallel utilization, so this is expected to be faster, particularly for large
point grids. The actual benefit depends on the engine and calculation size.

For example, reuse existing farm results while calculating a flow slice:

    .. code-block:: python

        slice_results = algo.calc_outputs(
            outputs="slice_data_xy_states",
            farm_results=farm_results,
        )

Point-result outputs and their writers do not define evaluation coordinates.
Supply ``point_results`` to reuse existing data, or combine them with an output
that requests points. The ``point_calculator`` preset likewise needs configured
``points`` or supplied point data. When passing coordinates directly to the
algorithm, use ``algo.calc_points(farm_results, points, outputs=...)``, which
also chunks in both dimensions.

.. _farm-result-outputs:

Farm results
------------

The :class:`FarmResultsOutput <foxes.models.outputs.farm_results.FarmResultsOutput>` model
collects farm data into an ``xarray.Dataset``. Without aggregation, physical
fields have dimensions ``(state, turbine)``, preserving the input state order
and the original turbine order in the wind farm.

The constructor parameters ``aggregate_turbines`` and ``aggregate_states``
control spatial grouping and weighted state reduction. Extensive quantities
such as power are summed over turbines, while intensive quantities are
averaged. See the class API for variable selection and aggregation details.

Farm variable sets
^^^^^^^^^^^^^^^^^^

The model book provides the following named farm-variable selections.
``AMB_`` denotes the ambient value without wake effects, and ``weight`` stores
the statistical state weights.

.. list-table::
    :header-rows: 1
    :widths: 15 50 35

    * - Name
      - Variables
      - Usage
    * - ``default``
      - ``X``, ``Y``, ``H``, ``D``, ``WD``, ``YAW``, ``AMB_RHO``,
        ``AMB_REWS``, ``REWS``, ``AMB_TI``, ``TI``, ``AMB_P``, ``P``, ``weight``
      - Common layout, wind and power diagnostics; selected by ``farm_results``.
    * - ``all``
      - Every available farm field; the list depends on the configured models.
      - Retain full datasets or additional model-specific fields.
    * - ``calc``
      - ``X``, ``Y``, ``H``, ``D``, ``WD``, ``YAW``, ``CT``,
        ``AMB_REWS``, ``REWS``, ``AMB_TI``, ``TI``, ``weight``
      - Retain the standard farm inputs for subsequent point calculations.
    * - ``P``
      - ``AMB_P``, ``P``, ``weight``
      - Compare ambient and waked turbine power.
    * - ``wsP``
      - ``AMB_REWS``, ``REWS``, ``AMB_P``, ``P``, ``weight``
      - Relate rotor-equivalent wind speeds to turbine power.
    * - ``wstiP``
      - ``AMB_REWS``, ``REWS``, ``AMB_TI``, ``TI``, ``AMB_P``, ``P``, ``weight``
      - Include turbulence intensity in wind-speed and power analysis.
    * - ``rhowstiP``
      - ``AMB_RHO``, ``AMB_REWS``, ``REWS``, ``AMB_TI``, ``TI``,
        ``AMB_P``, ``P``, ``weight``
      - Include ambient air density as well as wind, turbulence and power.
    * - ``analysis``
      - ``CAP``, ``AMB_CAPF``, ``CAPF``, ``EFF``, ``weight``
      - Evaluate nominal capacity, ambient/waked capacity factors and wake efficiency.

Append the set name to the result family, preserving its case. For example,
``farm_results_wsP`` selects wind-speed and power fields, while
``cluster_results_analysis_mean`` combines the analysis fields with cluster
aggregation and weighted state means. The same selections apply to writers,
for example ``farm_results_P_write_month`` for monthly power-result files.
``farm_results_default`` is equivalent to ``farm_results``.

The ``calc`` set includes thrust coefficients, which are not part of
``default``. Point calculations with additional model-specific farm inputs may
need an explicit variable list or the ``all`` set instead. The ``analysis``
selection requests the derived turbine metrics automatically.

For example:

    .. code-block:: python

        farm_results = algo.calc_farm(outputs="farm_results_calc")
        power_results = algo.calc_farm(outputs="farm_results_P")

.. _point-result-outputs:

Point results
-------------

The :class:`PointResultsOutput <foxes.models.outputs.point_results.PointResultsOutput>` model
collects flow quantities at evaluation points. The default ``point_results``
model-book entry selects wind speed, wind direction, turbulence intensity,
air density, their ambient counterparts and state weights.
``point_results_all`` collects all available point fields.

The default dataset returned by ``algo.calc_points()`` has physical fields on
``(state, point)`` and retains the supplied point order. The constructor
parameters ``aggregate_points`` and ``aggregate_states`` select spatial
averaging and weighted state reduction.

With ``algo.calc_points(..., outputs=...)``, the supplied ``points`` or
``point_results`` determine the evaluation locations; points defined by the
selected outputs are ignored. In contrast, ``algo.calc_outputs(outputs=...)``
uses the points defined by the outputs, unless ``point_results`` is supplied.
In that case, the existing point data and their coordinates take precedence
and are reused without scientific point recalculation.

Point variable sets
^^^^^^^^^^^^^^^^^^^

Point-variable selections are independent of the farm-variable sets:

.. list-table::
    :header-rows: 1
    :widths: 15 50 35

    * - Name
      - Variables
      - Usage
    * - ``default``
      - ``WD``, ``WS``, ``TI``, ``RHO``, ``AMB_WD``, ``AMB_WS``,
        ``AMB_TI``, ``AMB_RHO``, ``weight``
      - Collect standard wind fields, their ambient counterparts and state weights.
    * - ``all``
      - Every available point field; the list depends on the configured models.
      - Retain additional fields produced by point models.

``point_results`` and ``point_results_default`` select the same fields.
Use ``point_results_all`` to retain all available point fields, or combine a set
with reductions and writer policies, for example
``point_results_all_mean`` or ``point_results_all_write_month``.

For example, using farm results and evaluation coordinates:

    .. code-block:: python

        point_results = algo.calc_points(
            farm_results,
            points,
            outputs="point_results_all",
        )

Named variable sets are model-book choices, not values for the ``variables``
constructor parameter. When constructing
:class:`FarmResultsOutput <foxes.models.outputs.farm_results.FarmResultsOutput>`
or :class:`PointResultsOutput <foxes.models.outputs.point_results.PointResultsOutput>`
directly, pass a list of physical variable names, or ``None`` for all available
fields.

.. _output-files-and-figures:

Files and figures
-----------------

Output models can also write results to files or create figures. Typical file
and table outputs are:

* :class:`FarmResultsCountWriter <foxes.models.outputs.farm_results_writers.FarmResultsCountWriter>`,
  :class:`FarmResultsChunkWriter <foxes.models.outputs.farm_results_writers.FarmResultsChunkWriter>` and
  :class:`FarmResultsPeriodWriter <foxes.models.outputs.farm_results_writers.FarmResultsPeriodWriter>`:
  Write farm results to NetCDF files by state count, engine chunk or calendar period.
* :class:`PointResultsCountWriter <foxes.models.outputs.point_results_writers.PointResultsCountWriter>`,
  :class:`PointResultsChunkWriter <foxes.models.outputs.point_results_writers.PointResultsChunkWriter>` and
  :class:`PointResultsPeriodWriter <foxes.models.outputs.point_results_writers.PointResultsPeriodWriter>`:
  Write point results to NetCDF files.
* :class:`StateTurbineTable <foxes.models.outputs.state_turbine_table.StateTurbineTable>`:
  Create tables of state and turbine data.

For plotting and animation outputs, see the
:download:`outputs example notebook <notebooks/outputs.ipynb>`.

These outputs use the same selection mechanism as result datasets. Configure
file names, plot variables and other options in the output constructor.
For headless scripts, set ``show=False`` on plotting outputs to avoid opening
interactive windows.
