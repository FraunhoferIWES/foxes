.. _outputs:

Outputs
=======

The outputs of *foxes* runs depend on the choices by the user:

* :ref:`Farm results <farm-result-outputs>`: Data for each state and turbine,
  optionally combined over turbines or states.
* :ref:`Point results <point-result-outputs>`: Flow quantities at evaluation
  points.
* :ref:`Files and figures <output-files-and-figures>`: Result files, tables,
  wind farm layouts, wind roses, flow slices and animations.

All concrete output models are implemented in the
:doc:`foxes.models.outputs <_autoapi/foxes/models/outputs/index>` sub-package.
For a visual introduction, see the :doc:`outputs example <notebooks/outputs>`.

.. _output-selection:

Calculation functions
---------------------

Output models are stored in the so-called
:class:`ModelBook <foxes.models.model_book.ModelBook>` object under a name
string, in the same way as the :doc:`Models <models>`. Select them through the
``outputs`` parameter of the calculation functions:

.. list-table::
    :header-rows: 1
    :widths: 20 40 40

    * - Entry point
      - Inputs and work
      - Output behavior
    * - ``calc_farm``
      - Calculate turbine/farm physics from the configured states and turbines.
      - Execute outputs on new farm chunks; output-requested points are evaluated
        in a loop over bounded point blocks within each state chunk.
    * - ``calc_points``
      - Calculate at caller-supplied ``points`` using provided ``farm_results``,
        or reuse optionally supplied ``point_results``.
      - Ignore output-requested grids. The supplied locations/results determine
        where compatible point outputs run.
    * - ``calc_outputs``
      - Reuse supplied results and calculate missing farm/point inputs when needed.
      - Honor output-requested points unless precomputed point results are
        supplied; metadata-only outputs require no scientific chunks.

The parameter accepts a model-book name, an output model object, or a list of
either. For example, the default farm output returns an ``xarray.Dataset``:

    .. code-block:: python

        farm_results = algo.calc_farm(outputs="farm_results")

To choose individual variables, pass a configured output object. For example,
power with and without wake effects can be selected as

    .. code-block:: python

        import foxes.variables as FV
        from foxes.models.outputs import FarmResultsOutput

        power_results = algo.calc_farm(
            outputs=FarmResultsOutput(variables=[FV.P, FV.AMB_P]),
        )

Note that ``outputs`` selects models, not physical variable names. A single
model returning one result returns that value directly. Multiple outputs return
a dictionary keyed by result names; ``ret_dict=True`` retains this dictionary
also for a single output. Writers alone return an empty dictionary.

For outputs that evaluate flow at points, prefer ``algo.calc_outputs()`` with
existing farm results. This allows the engine to distribute both state and
point chunks. See the :doc:`Overview <overview>` for the calculation workflow.

Variable sets
-------------

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
        ``AMB_REWS``, ``REWS``, ``AMB_TI``, ``TI``, ``CT``, ``AMB_P``, ``P``, ``weight``
      - Common diagnostics and standard inputs for subsequent point calculations;
        selected by ``farm_results``.
    * - ``all``
      - Every available farm field; the list depends on the configured models.
      - Retain full datasets or additional model-specific fields.
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

Append the set name to the result model name, preserving its case, e.g.
``farm_results_wsP`` or ``point_results_all``. Results models select ``default``
if the set name is omitted. Use ``_amb`` or ``_waked`` before the set name to
filter ambient or waked variables, for example ``farm_results_amb_wsP``.

Named sets are model-book choices. When constructing a results model directly,
pass a list of physical variable names to ``variables``, or ``None`` for all
available fields.

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

Point-result outputs and their writers do not define evaluation coordinates.
Supply existing ``point_results``, combine them with an output that requests
points, or pass coordinates to ``algo.calc_points()``. The ``point_calculator``
preset likewise needs configured points or supplied point data.

.. _farm-result-outputs:

Farm results
------------

The :class:`FarmResultsOutput <foxes.models.outputs.farm_results.FarmResultsOutput>`
model returns data for each state and turbine as an ``xarray.Dataset``.
Without aggregation, physical fields have dimensions ``(state, turbine)``,
in the original turbine order. Known variables carry their units.

The model-book name ``farm_results_analysis`` requests turbine performance
metrics, calculated automatically. Note that ``farm_results_all`` retains
available fields; it does not request additional metrics.

Results can also be combined over turbines. The model book provides the
following families, each accepting the farm variable sets listed above:

* ``farm_results``: Keep individual turbines.
* ``farms_results``: Combine turbines belonging to the same wind farm.
* ``cluster_results``: Combine turbines belonging to the same cluster.
* ``total_results``: Combine all turbines.

Append ``_mean`` for weighted means over states, for example
``cluster_results_analysis_mean``. Power and capacity are summed over turbines,
while wind speeds are averaged. Efficiency and capacity factors are calculated
from the combined power values. These choices can also be configured through
``aggregate_turbines`` and ``aggregate_states`` on a results model.

.. _point-result-outputs:

Point results
-------------

The :class:`PointResultsOutput <foxes.models.outputs.point_results.PointResultsOutput>`
model returns flow quantities at evaluation points. The default dataset from
``algo.calc_points()`` has physical fields on ``(state, point)`` and retains
the supplied point order.

For example, wind speed and other standard flow quantities at two points can
be calculated as

    .. code-block:: python

        import numpy as np

        farm_results = algo.calc_farm(outputs="farm_results_all")
        point_results = algo.calc_points(
            farm_results,
            np.array([[400.0, 0.0, 90.0], [600.0, 0.0, 90.0]]),
            outputs="point_results",
        )

Here ``farm_results_all`` retains farm fields needed by the wake calculation.
With ``calc_points()``, the supplied points determine the evaluation locations.
With ``calc_outputs()``, outputs can request their own points instead.
Supplied ``point_results`` take precedence and are reused without recalculation.

Append ``_aggregate_all`` to average over points, and ``_mean`` for weighted
means over states, e.g. ``point_results_all_aggregate_all_mean``. Note that
``_all`` selects variables, whereas ``_aggregate_all`` combines points.
These choices correspond to ``aggregate_points`` and ``aggregate_states``
on a results model.

.. _output-files-and-figures:

Files and figures
-----------------

Writing results
^^^^^^^^^^^^^^^

Farm and point results can be written to NetCDF files by appending a writer
suffix to a model-book name:

* ``_write_n<count>``: Write a file for every ``count`` states, e.g.
  ``farm_results_P_write_n100``.
* ``_write_chunks``: Write one file for each calculation chunk.
* ``_write_day``, ``_write_week``, ``_write_month`` or ``_write_year``:
  Write calendar-period files, requiring chronological datetime states.

Variable sets, filters and aggregation can be combined with writing, e.g.
``cluster_results_amb_analysis_mean_write_month`` for monthly cluster summaries.
Count writers without a variable-set name write all available fields; chunk
and calendar writers use ``default``.

Select a results model alongside the writer to keep a dataset in memory:

    .. code-block:: python

        results = algo.calc_farm(
            outputs=["farm_results", "farm_results_P_write_n100"],
        )

For custom filenames, directories or NetCDF options, use a writer object with
``base_name``, ``out_dir`` and ``write_nc_pars``. The writer classes are listed
in the :doc:`output models API <_autoapi/foxes/models/outputs/index>`.

Plots and flow slices
^^^^^^^^^^^^^^^^^^^^^

Typical model-book choices for figures, tables and flow data are:

* ``farm_layout``: Plot turbine positions.
* ``rose_plot``: Plot a wind rose at a turbine.
* ``states_rose_plot``: Plot an ambient wind rose at a chosen location.
* ``support_rose_plot``: Plot the native wind distributions of binned states.
* ``wind_rose_bin_plot``: Plot a turbine variable, such as power, in wind bins.
* ``state_turbine_map`` and ``state_turbine_table``: Show farm variables
  as a heat map or table.
* ``point_calculator``: Return data at configured points, optionally averaged
  over states.
* ``rotor_point_plot``: Show the rotor model's evaluation points.
* ``turbine_type_curves``: Plot power and thrust curves from a configured
  wind-speed scan.
* ``slice_data`` and ``slices_data``: Return flow data on one or more planes.
* ``flow_plot``: Plot a flow-field slice.

The slice families provide the name templates
``slice_data_<plane>_<reduction>``, ``slices_data_<plane>_<reduction>`` and
``flow_plot_<plane>_<reduction>``. Here ``plane`` is ``xy``, ``xz`` or ``yz``,
and ``reduction`` is ``states`` or ``mean``.

Configure a model-book output before calculation, or pass a configured output
object. For example, a mean wind-speed plot on a vertical plane is obtained as

    .. code-block:: python

        output = algo.mbook.outputs["flow_plot_xz_mean"]
        output.grid_pars = {"n_img_points": (30, 20)}
        output.show = False
        figure = algo.calc_outputs(outputs=output, farm_results=farm_results)

Static plotting presets display their figures. Set ``show=False`` for scripts
without an interactive display; this is already the default when constructing
plotting classes directly. For flow plots and roses, ``plot=False`` returns
numerical data instead. ``nofig=True`` skips image outputs while retaining
explicit numerical outputs and writers.

Animations
^^^^^^^^^^

The model book provides ``create_gif`` and ``create_mp4`` for flow animations.
The variants ``create_turbine_gif`` and ``create_turbine_mp4`` add synchronized
turbine traces below the flow field. ``chunk_gif_<chunk>`` and
``chunk_mp4_<chunk>`` select a single state chunk.

For example, continuing with the farm results above:

    .. code-block:: python

        animation = algo.mbook.outputs["create_gif"]
        animation.base_name = "flow.gif"
        animation.grid_pars = {"n_img_points": (30, 20)}
        animation.fps = 5
        algo.calc_outputs(outputs=animation, farm_results=farm_results)

This writes ``flow_WS.gif`` for wind speed. MP4 output requires an installed
ffmpeg encoder. Animation presets save files without opening a display window;
set ``show=True`` to display them as well.
