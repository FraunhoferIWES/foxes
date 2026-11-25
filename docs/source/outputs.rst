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

Output classes
--------------

Output classes are useful when you need constructor options that are not
available through a preset. Configure an instance, then pass it to
``calc_farm``, ``calc_points`` or ``calc_outputs``.

.. _farm-result-outputs:

Farm results
^^^^^^^^^^^^

Use :class:`FarmResultsOutput
<foxes.models.outputs.farm_results.FarmResultsOutput>` to collect selected
fields as an ``xarray.Dataset``. Without aggregation, physical fields have
dimensions ``(state, turbine)`` and retain the original turbine order. Known
variables include their units.

.. _point-result-outputs:

Point results
^^^^^^^^^^^^^

Use :class:`PointResultsOutput
<foxes.models.outputs.point_results.PointResultsOutput>` with ``calc_points``
to collect flow fields at supplied coordinates:

    .. code-block:: python

        import numpy as np
        import foxes.variables as FV
        from foxes.models.outputs import PointResultsOutput

        points = np.array([[400.0, 0.0, 90.0], [600.0, 0.0, 90.0]])
        point_results = algo.calc_points(
            farm_results,
            points,
            outputs=PointResultsOutput(variables=[FV.WS]),
        )

The result has dimensions ``(state, point)`` and preserves the supplied point
order.

.. _output-files-and-figures:

Other output classes
^^^^^^^^^^^^^^^^^^^^

The :doc:`output model API <_autoapi/foxes/models/outputs/index>` documents
constructor options for the other output classes:

* ``FarmResultsCountWriter``, ``FarmResultsChunkWriter`` and
  ``FarmResultsPeriodWriter`` write farm results; ``PointResultsCountWriter``,
  ``PointResultsChunkWriter`` and ``PointResultsPeriodWriter`` write point
  results.
* ``PointCalculator``, ``SliceData`` and ``SlicesData`` collect configured
  point or flow-slice data. ``FlowPlots2D`` plots a flow slice.
* ``FarmLayoutOutput``, ``RosePlotOutput``, ``StatesRosePlotOutput``,
  ``SupportWindRosePlotOutput`` and ``WindRoseBinPlot`` create layout and
  wind-rose figures.
* ``StateTurbineMap``, ``StateTurbineTable``, ``RotorPointPlot`` and
  ``TurbineTypeCurves`` provide other common farm and turbine views.

.. _model-book-outputs:

Model-book outputs
------------------

Use a model-book name for a ready-made output configuration. Names are
selectors for output models, not variable names. For example:

    .. code-block:: python

        farm_results = algo.calc_farm(outputs="farm_results")
        power_results = algo.calc_farm(outputs="farm_results_P")

Entries can also be retrieved from ``algo.mbook.outputs`` and configured before
use. For example, configure a flow plot and run it using existing farm results:

    .. code-block:: python

        output = algo.mbook.outputs["flow_plot_xz_mean"]
        output.grid_pars = {"n_img_points": (30, 20)}
        output.show = False
        figure = algo.calc_outputs(outputs=output, farm_results=farm_results)

Farm result entries
^^^^^^^^^^^^^^^^^^^

The farm-result families differ in how they combine turbines:

* ``farm_results`` keeps results for each turbine.
* ``farms_results`` combines turbines with the same wind-farm membership.
* ``cluster_results`` combines turbines with the same cluster membership.
* ``total_results`` combines all turbines.

The grouped families sum power and capacity and average wind speeds; efficiency
and capacity factors are calculated from the combined power. Append ``_mean``
for weighted means over states, for example ``cluster_results_analysis_mean``.

Farm variable sets
""""""""""""""""""

Append a variable-set name to a farm-result entry. The default set is selected
when omitted. ``AMB_`` fields are ambient values without wake effects; ``weight``
contains state weights.

.. list-table::
    :header-rows: 1
    :widths: 15 55 30

    * - Name
      - Variables
      - Use
    * - ``default``
      - ``X``, ``Y``, ``H``, ``D``, ``WD``, ``YAW``, ``AMB_RHO``,
        ``AMB_REWS``, ``REWS``, ``AMB_TI``, ``TI``, ``CT``, ``AMB_P``, ``P``,
        ``weight``
      - Standard results and inputs for later point calculations.
    * - ``all``
      - Every available farm field; depends on configured models.
      - Keep additional model-specific fields.
    * - ``P``
      - ``AMB_P``, ``P``, ``weight``
      - Compare ambient and waked power.
    * - ``wsP``
      - ``AMB_REWS``, ``REWS``, ``AMB_P``, ``P``, ``weight``
      - Relate rotor-equivalent wind speeds to power.
    * - ``wstiP``
      - ``AMB_REWS``, ``REWS``, ``AMB_TI``, ``TI``, ``AMB_P``, ``P``,
        ``weight``
      - Include turbulence intensity.
    * - ``rhowstiP``
      - ``AMB_RHO``, ``AMB_REWS``, ``REWS``, ``AMB_TI``, ``TI``,
        ``AMB_P``, ``P``, ``weight``
      - Include ambient density as well as wind, turbulence and power.
    * - ``analysis``
      - ``CAP``, ``AMB_CAPF``, ``CAPF``, ``EFF``, ``weight``
      - Include capacity and performance metrics.

For example, ``farm_results_wsP`` collects the ``wsP`` fields. Add ``_amb`` or
``_waked`` before the set name to select only ambient or waked variables, as in
``farm_results_amb_wsP``. The ``analysis`` set requests capacity and performance
metrics; ``farm_results_all`` only retains fields already available.

Point result entries
^^^^^^^^^^^^^^^^^^^^

``point_results`` collects the default point-variable set and
``point_results_all`` collects every available point field. Point-variable sets
are independent of farm-variable sets:

.. list-table::
    :header-rows: 1
    :widths: 15 55 30

    * - Name
      - Variables
      - Use
    * - ``default``
      - ``WD``, ``WS``, ``TI``, ``RHO``, ``AMB_WD``, ``AMB_WS``,
        ``AMB_TI``, ``AMB_RHO``, ``weight``
      - Standard flow fields and state weights.
    * - ``all``
      - Every available point field; depends on configured models.
      - Keep additional fields from point models.

Append ``_aggregate_all`` to combine points and ``_mean`` to calculate weighted
state means, for example ``point_results_all_aggregate_all_mean``. Here ``_all``
selects variables, while ``_aggregate_all`` combines points. Point collectors
and writers do not define coordinates: supply points to ``calc_points``, use
precomputed point results, or choose an output that requests its own points.
``point_calculator`` likewise needs configured points or supplied point data.

Writing results
^^^^^^^^^^^^^^^

Append a writer suffix to a farm- or point-result entry to write NetCDF files:

* ``_write_n<count>`` writes a file for every ``count`` states, e.g.
  ``farm_results_P_write_n100``.
* ``_write_chunks`` writes one file per calculation chunk.
* ``_write_day``, ``_write_week``, ``_write_month`` or ``_write_year`` writes
  calendar-period files; states must have chronological datetime coordinates.

For example, ``cluster_results_amb_analysis_mean_write_month`` writes monthly
cluster summaries. Variable sets, filters and aggregation can be combined with
writers. Count writers without a variable-set name write all available fields;
chunk and calendar writers use ``default``. Select a collector alongside a
writer to keep results in memory:

    .. code-block:: python

        results = algo.calc_farm(
            outputs=["farm_results", "farm_results_P_write_n100"],
        )

Plotting and flow outputs
^^^^^^^^^^^^^^^^^^^^^^^^^

Common model-book entries include:

* ``farm_layout`` for a farm layout; ``state_turbine_map`` and
  ``state_turbine_table`` for farm fields by state and turbine.
* ``rose_plot``, ``states_rose_plot``, ``support_rose_plot`` and
  ``wind_rose_bin_plot`` for wind-rose plots.
* ``point_calculator``, ``rotor_point_plot`` and ``turbine_type_curves`` for
  point data and turbine views.
* ``slice_data_<plane>_<reduction>``,
  ``slices_data_<plane>_<reduction>`` and
  ``flow_plot_<plane>_<reduction>`` for flow slices. ``plane`` is ``xy``,
  ``xz`` or ``yz``; ``reduction`` is ``states`` or ``mean``.
* ``create_gif`` and ``create_mp4`` for flow animations, with
  ``create_turbine_gif`` and ``create_turbine_mp4`` adding turbine traces.

Model-book plotting entries display figures by default. Set ``show=False`` for
headless scripts. For flow plots and roses, ``plot=False`` returns numerical
data instead; ``nofig=True`` skips image outputs while retaining explicit
numerical outputs and writers.
