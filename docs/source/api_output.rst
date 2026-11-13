Output models
=============
For a runnable introduction, see :doc:`the practical output guide <notebooks/outputs>`.
It demonstrates farm and point results, automatic turbine evaluation, aggregation,
point probes, plots, writers and animation with small synthetic inputs.

Adopted outputs derive from ``foxes.core.Output`` and are exported from
``foxes.models.outputs``. Engine workers calculate chunk-local NumPy payloads;
the output's ``update_results()`` and ``finalize_results()`` hooks unify them.
``OutputList`` composes models in declaration order and merges their required
input variables.

Processing supplied results
---------------------------
``Output.requires_chunk_calculation`` defaults to ``True``. Outputs override
this read-only property with ``False`` when their initial-result and
parent-side finalization hooks can process supplied data directly.
``calc_outputs`` never calls ``calc_farm`` for supplied farm fields without
requested point calculations; the engine runs only the required output work.

``StateTurbineTable`` without aggregation formats supplied farm results directly
on the parent, including field selection, power-unit normalization, renaming,
transposition and optional NetCDF writing. It does not dispatch chunks or copy
unchanged scientific arrays, and does not mutate the supplied dataset.
State selection is applied before the initial-result hook. Tables configured
with ``aggregate_turbines`` or ``aggregate_states`` retain chunk reductions.

``OutputList`` requests chunks when any child needs them, but skips parent-only
children during output-only chunk evaluation. Every child retains the normal
initialization, running, finalization and restoration lifecycle. Fresh scientific
calculations still collect every selected output from chunks, regardless of
this property. ``model_results=None`` identifies output-only chunks.

Constructor-defined consumers
-----------------------------
``FarmLayoutOutput``, ``SliceData``, ``SlicesData``,
``FlowPlots2D``, ``RosePlotOutput``, ``StatesRosePlotOutput`` and
``WindRoseBinPlot`` are exported only from ``foxes.models.outputs``.
Configure the entire operation before execution; do not pass an algorithm or
results to an output constructor or call dataset-bound consumer methods.
Constructors retain configuration only, not large scientific data.

The public ``outputs`` selector is unchanged: model-book strings, instances
and lists of either are supported by ``calc_farm``, ``calc_points`` and
``calc_outputs``. For example:

.. code-block:: python

    from foxes.models.outputs import FarmResultsOutput, SliceData

    with foxes.Engine.new("numpy", chunk_size_states=64, chunk_size_points=500):
        slice_data = algo.calc_farm(outputs=SliceData(
            plane="xz", variables=[FV.WS], reduction="mean",
            grid_pars={"n_img_points": (30, 20)},
        ))
        farm_results = algo.calc_farm(outputs=FarmResultsOutput(
            variables=[FV.P, FV.AMB_P, FV.CAP, FV.CAPF, FV.EFF, FV.YLD],
        ))

All three entry points accept ``ret_dict=True`` to retain the result-name mapping
even when one selected output returns exactly one value. The default remains
the unwrapped value; composed outputs, writers and disabled collection retain
their existing schemas. ``Iterative.calc_farm`` preserves only its final output
mapping, leaving convergence datasets and optional downwind-order history intact.

.. code-block:: python

    results = algo.calc_farm(outputs=["farm_results"], ret_dict=True)
    for key, data in results.items():
        print(f"{key}:\n{data}")
    outputs = algo.calc_outputs(
        outputs="state_turbine_table",
        farm_results=results.get("farm_results"),
        ret_dict=True,
    )

Derived farm fields are evaluated automatically by ``TurbineEvaluation``
through output preparation. No turbine-model registration is needed.
``variables=None`` collects available fields without adding derived metrics.
The ``farm_results_analysis`` collector selects capacity, capacity factors,
efficiency and weights. Metrics initially retain ``(state, turbine)``
dimensions; configure the collector's independent spatial/state reductions
described below for aggregated summaries.

``SliceData`` returns NumPy, pandas or xarray data according to ``data_format``.
``SlicesData(positions=[...], plane=...)`` combines parallel planes into one
xarray dataset and writes it once. ``FlowPlots2D`` selects ``var``, ``plane``,
``reduction`` and ``plot`` in its constructor: plots return one mean figure or
a rendered list of state figures; numerical mode returns
``(parameters, data, grid)`` or ``(parameters, data, states, grid)``.
``ret_data=True`` adds the formatted data to a plotted result.
``FlowAnimation`` derives from ``SliceData``; ``SliceData``, ``SlicesData`` and
``FlowPlots2D`` do not inherit animation settings or validation. Shared grids
appear under model-private ``slice_points`` keys, with numerical buffers named
``slice_chunks`` and optional loaded selection masks named ``slice_selection``.
Flow-plot data uses the same point ordering and weighted reduction as slices.
Layout outputs select an operation such as ``get_figure`` or ``write_csv``
and pass its options in ``parameters``.
Rose outputs configure bins, location and rendering; ``plot=False`` returns a
frequency dataset and ``plot=True`` returns ``(axes, dataset)``.
Wind-rose sectors retain meteorological bearings: north is up and east is right.
``StatesRosePlotOutput(point=(x, y, z))`` evaluates the executing algorithm's
states at that location without constructor-time calculations.

Point requests use ``get_calc_points(algo)`` after initialization. Farm runs
evaluate bounded point slices inside each state chunk; ``calc_outputs`` delegates
both chunk axes to the point engine. Composed outputs must request identical
points. ``calc_points`` ignores output requests and consumes its ``points`` or
``point_results`` instead. Precomputed datasets retain ``X``, ``Y`` and ``H``
coordinates and are not recalculated.

Displaying plots and animations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Plotting constructors default to ``show=False``: they create/return figures or
save animations without explicitly opening a display window. Set ``show=True``
on ``FarmLayoutOutput``, ``FlowPlots2D``, rose outputs or ``FlowAnimation``
(including ``FlowTurbineAnimation``) to call blocking Matplotlib display during
parent-side finalization. Files are saved before display. Animations play at
their configured ``fps`` while the window is open, then release their figure;
figures/animations are never stored on running output models.

Static model-book plotting presets and factories instead enable ``show=True``,
including ``flow_plot_<plane>_<reduction>``. Animation presets such as
``create_gif``, ``create_mp4``, their turbine variants and chunk-animation
factories keep ``show=False``. Disable display before executing static plotting
presets in scripts and headless workflows:

.. code-block:: python

    algo.mbook.outputs["flow_plot_xy_mean"].show = False
    result = algo.calc_farm(outputs="flow_plot_xy_mean")

``plot=False`` selects numerical results without display. ``nofig=True`` disables
image-producing modes before initialization, field preparation and point-grid
requests, including layout plot files and animations. Disabled outputs add no
result; selecting only disabled outputs returns ``{}``, not numerical plot data.
Nested ``OutputList`` instances propagate suppression without mutating children.
Numerical modes, data-only layout operations and writers remain enabled.
Mark numerical probes consumed exclusively by downstream plots with
``image_only=True`` to disable their calculations with ``nofig`` too.
Sequential animation plugins apply the same check before evaluating step points.
Data-only layout operations, numerical outputs and returned frame generators
do not force display. Showing windows or playing animations requires an
interactive Matplotlib backend; FOXES does not select a backend or silence
Matplotlib's headless-backend warnings.
Tests never enable ``show``, including mocked checks.

Animation diagnostics
~~~~~~~~~~~~~~~~~~~~~
``FlowAnimation`` and ``FlowTurbineAnimation`` accept ``verbosity=None`` to
inherit the executing algorithm's level, or an explicit level:

* ``0``: no animation diagnostics.
* ``1``: announce creation.
* ``2``: additionally announce the output filename when writing begins and
  completion only after a successful write; include frame/point counts,
  encoder choice and frames per second.
* ``3``: additionally report each composed and encoded frame.

Animation examples use their ``-v``/``--verbosity`` level directly, independently
of reduced library verbosity. Their default is ``2``; ``run_all.py`` explicitly
passes ``-v 0`` unless configured otherwise. Figure-less Python examples configure ``nofig`` on selected outputs; the
framework skips animation writers and image-only point calculations.
Explicit numerical outputs remain enabled.

Compact calculation payloads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``calculate()`` returns only data required by the final output, under a private
model key. Means are reduced on workers, not collected as full state histories:

* Mean slices, multi-plane slices and mean flow plots return weighted sums of
  shape ``(1, n_points)`` per variable and point tile. Global weights are not
  renormalized per chunk or selected subset. Raw weights and state frames are
  not transported. ``ret_states=True`` additionally retains selected labels.
  Explicit ``aggregate_states=True`` instead returns per-point sufficient
  statistics, normalizes selected-state weights after merging and circularly
  averages directions; it does not change ordinary ``reduction="mean"`` sums.
* Wind roses return bin accumulators rather than samples. Weighted means combine
  numerators and denominators globally before division and thresholding.
* Layouts return only the selected state geometry and reduced colour values.
  Algorithm-owned layouts without scientific colours return no worker arrays.
  Negative state/point indices use bounded trailing candidates for final selection.

Flow plots collect their selected variable, adding wind speed/direction only for
configured quivers and farm geometry only for rotor overlays. Static point
coordinates have a singleton state axis in worker payloads; final public point
datasets restore full dimensions. State-resolved datasets, writers and animations
retain the requested state fields because their final results require them.
Aggregated collectors and writers eliminate state histories on workers.
Count/calendar writers partition each worker chunk into exact file intervals
and return only per-entity weighted statistics for those intervals. Spatial-only
point aggregation returns target sums/counts rather than averages of tile means.
These payload reductions do not alter the full-shaped scientific data containers.

YAML execution
~~~~~~~~~~~~~~
Configured YAML entries specify a class name or model-book preset in
``output_type``, without ``functions``. Class entries accept constructor options;
presets are copied and accept no constructor overrides. Customize a preset in
``model_book.outputs`` or use its class name instead. They run through
``calc_outputs``; ``with_engine`` selects the
execution phase and defaults to true. ``result_labels`` stores the result, and
nested ``$name`` references can supply previously created axes.
Mutually exclusive ``states_sel`` (labels) and ``states_isel`` (integer positions,
including negative positions) select inputs before aggregation. Both support
stored ``$name`` references.
YAML-only ``plt`` calls are handled directly by the YAML reader, not output models.

.. code-block:: yaml

    outputs:
      - output_type: farms_results_all_mean
        states_isel: [0, 2]
        result_labels: $summary

Model-book consumer presets
---------------------------
The adopted consumers are available through ``ModelBook.outputs`` without
constructing grids, loading scientific data or running calculations:

====================== =========================== ============================
Name                   Output class                Preset
====================== =========================== ============================
``farm_layout``        ``FarmLayoutOutput``        Layout figure
``rose_plot``          ``RosePlotOutput``          Farm wind rose, turbine zero
``states_rose_plot``   ``StatesRosePlotOutput``    Rose at ``(0, 0, 100)`` metres
``wind_rose_bin_plot`` ``WindRoseBinPlot``         Power in wind-speed bins
``slice_data``         ``SliceData``               State-resolved XY data
``slices_data``        ``SlicesData``              XY plane at 100 metres
``flow_plot``          ``FlowPlots2D``              Mean XY wind-speed figure
====================== =========================== ============================

Additional presets are ``state_turbine_map`` (power heat map),
``state_turbine_table`` (power/rotor-speed table),
``point_calculator`` (supplied point wind speed),
``rotor_point_plot`` (executing rotor metadata), and ``turbine_type_curves``
(power/thrust curves from an explicit scan). Configure arbitrary point locations
or curve scan states on the appropriate input before execution.
``plt`` is deliberately not a model-book entry; it is YAML-only.

``support_rose_plot`` selects ``SupportWindRosePlotOutput`` for algorithms using
``BinnedFieldData`` or ``BinnedPointCloudData``. It renders all native support
points using their original ``WS``/``WD`` bin edges and percentage frequencies,
without interpolation or scientific recalculation. Set ``plot=False`` for only
the ``(point, WD, WS)`` frequency dataset. Plotting returns ``(figure, data)``;
``plot_pars`` accepts ``ncols``, ``figsize``, ``title``, ``fig`` and polar ``axs``.
Use ``to_file`` to save the figure during finalization. ``nofig=True`` disables
image mode before initialization; it does not disable explicit numerical mode.

Outputs without supplied scientific results or requested calculation points
still enter the normal initial-result, running and finalization lifecycle.
They do not dispatch scientific chunks.

The binned power preset uses speed edges ``(0, 5, 10, 15, 20, 25)`` m/s.
The slice families also support ``slice_data_<plane>_<reduction>``,
``slices_data_<plane>_<reduction>`` and
``flow_plot_<plane>_<reduction>``, with ``plane`` in ``xy``, ``xz``, ``yz``
and ``reduction`` in ``states``, ``mean``. Multi-plane presets use the single
normal-axis position ``100`` metres; change ``positions`` before execution
for multiple planes or different locations. Configure other preset attributes
before initialization, or supply a fully configured output instance directly.
For example:

.. code-block:: python

    algo.mbook.outputs["slice_data_xz_mean"].grid_pars = {
        "n_img_points": (30, 20),
    }
    data = algo.calc_farm(outputs="slice_data_xz_mean")

For a static mean XY flow figure use ``outputs="flow_plot_xy_mean"``.
Setting ``algo.mbook.outputs["flow_plot_xy_mean"].plot = False`` returns its
numerical data tuple instead; neither mode initializes an animation.

Implemented farm outputs
------------------------

* ``FarmResultsOutput`` collects one xarray dataset. Select fields with
  ``variables`` and filter ambient/ordinary fields with ``ambient_vars`` and
  ``waked_vars``. State/turbine coordinates are preserved even when the field
  selection is empty. Optional turbine, farm and cluster names are attached
  along the turbine axis. Uniform weights have only the state dimension.
* ``FarmResultsChunkWriter`` writes ``<base_name>_<chunk index:06d>.nc`` from
  each engine state chunk.
* ``FarmResultsCountWriter`` writes ``<base_name>_<file index:06d>.nc`` with
  exactly ``n_states_per_file`` states except for the final remainder. The
  positive integer limit is independent of engine chunk size.
* ``FarmResultsPeriodWriter`` writes calendar day, ISO week, month or year
  files. Datetime states must be chronological and contain no ``NaT``.
  Weeks begin on Monday; ``YYYY-Www`` uses the ISO year. Missing periods do not
  create empty files.

All writers apply ``out_dir`` and ``out_fname_fun`` and forward
``write_nc_pars`` to ``foxes.utils.write_nc``. For example, select the NetCDF
backend with ``{"nc_engine": "h5netcdf"}``, or enable lossless packing with
``{"pack": True}``. Buffered writers await submitted writes during finalization,
so worker failures propagate.

File writing is owned by these output models, not by engines. Select a writer
through ``outputs`` and compose it with ``FarmResultsOutput`` when an in-memory
dataset is also required. Engine backends only dispatch chunk calculations and
coordinate the output lifecycle.

Physical aggregation
~~~~~~~~~~~~~~~~~~~~
Farm collectors and every farm writer accept
``aggregate_turbines=None | "wind_farm" | "cluster" | "all"``.
Point collectors and every point writer accept ``aggregate_points=None | "all"``.
Both have independent ``aggregate_states=False``. The default retains the
original dimensions. Configure aggregation on an instance before execution;
no separate evaluation class or generic results writer is used.

Farm/cluster grouping uses original turbine identities and first-appearance
group order, including a reserved ``<unassigned>`` label for missing memberships.
Grouping replaces ``turbine`` with ``farm`` or ``cluster``; ``"all"`` removes
the spatial axis. Point grouping removes ``target``, retaining each
``target_point`` position in files.

Power, capacity, maximum power and annualized yield fields are summed over
turbines. Intensive fields use arithmetic means; wind/yaw directions use
circular means (NaN for cancelling vectors). Point fields are averaged.
Power, ambient power, capacity and maximum power are converted to the smallest
power unit used by the farm's initialized turbine types before any reduction.
For example, combining MW and kW types produces kW values, without splitting
groups by unit. Uniform farms retain their model's ``P_unit``. Yields stay in
GWh/a. Scientific model containers and private convergence history retain
native units.

Dataset units
~~~~~~~~~~~~~
Returned numerical datasets and written NetCDF files carry a ``units``
attribute on known scientific variables and coordinates, including points,
renamed tables and slices, roses and normalized rotor metadata:

.. code-block:: python

    results = algo.calc_farm(outputs="farm_results")
    power_unit = results["P"].attrs["units"]
    speed_unit = results["REWS"].attrs["units"]  # "m/s"

Power fields share one unit per variable, also recorded in dataset attribute
``power_unit``. Reprocessing collected farm fields uses their declared units
instead of converting them again as native turbine values. Metre coordinates
use ``m``; ratios, statistical weights and normalized rotor coordinates use
``1``; annualized yields use ``GWh/a``; rose frequency uses ``%``.
Slice coordinates/fields divided by non-unit normalization factors use ``1``.
Renaming and Sequential accumulation preserve metadata. Unknown custom fields
are not assigned speculative units.

State aggregation
~~~~~~~~~~~~~~~~~

``aggregate_states=True`` computes weighted means over selected states,
normalizing weights independently for every turbine or target/target-point
position, before spatial grouping. This also applies to partial selections
and each writer file independently. Efficiency and capacity factor are
recomputed from reduced power/capacity, never averaged. Zero power gives zero
ratios, nonzero power over zero gives positive infinity, and NaNs propagate
otherwise. Zero-weight NaN samples are ignored; negative/nonfinite weights
and zero total weight raise ``ValueError``.

Dataset attributes ``n_states``, ``n_turbines`` and ``n_points`` record scalar
input counts only when the corresponding dimension is aggregated. Counts cover
the entire selected result or file interval; chunk files describe their own
engine tile. No per-variable counts or weight-sum arrays are added. Existing
count/calendar file attributes remain available even without aggregation.

Workers retain per-entity sufficient statistics without a state dimension.
Exact count/calendar file boundaries can split or span engine chunks without
transporting individual samples. Chunk writers reduce only their supplied
engine tile, not targets/states belonging to other chunk files.
Raw and grouped collectors can be composed when their fixed public result
keys are distinct. Duplicate keys raise ``ValueError``.

Farm dataset keys are ``FC.FARM_RESULTS``, ``FC.FARMS_RESULTS``,
``FC.CLUSTER_RESULTS`` and ``FC.TOTAL_FARM_RESULTS`` for individual turbines,
wind farms, clusters and all turbines, respectively. With state reduction,
use ``FC.MEAN_FARM_RESULTS``, ``FC.MEAN_FARMS_RESULTS``,
``FC.MEAN_CLUSTER_RESULTS`` and ``FC.MEAN_TOTAL_FARM_RESULTS``. Point datasets use
``FC.POINT_RESULTS`` or ``FC.TOTAL_POINT_RESULTS`` for individual or aggregated
points, and ``FC.MEAN_POINT_RESULTS`` or ``FC.MEAN_TOTAL_POINT_RESULTS`` with
state reduction. Farm and point totals have distinct keys and can be composed.
Model-book families instead encode named turbine groups directly:
``farms_results`` selects ``aggregate_turbines="wind_farm"`` and
``cluster_results`` selects ``aggregate_turbines="cluster"``.
``total_results`` selects ``aggregate_turbines="all"``; ``farm_results`` retains
individual turbines. These grouped families do not append aggregation suffixes.
Dataset keys reflect grouping and state reduction, not field or filter presets.

These presets select each family's default fields. Every collector and writer
also combines optional ambient/waked filters and named fields with aggregation:

.. code-block:: text

    <family>[_amb|_waked][_<variables>][_aggregate_all][_mean]
        [_write_n<count>|_write_chunks|_write_<period>]

Families are ``farm_results``, ``farms_results``, ``cluster_results``,
``total_results`` and ``point_results``; ``_aggregate_all`` is available only for
``point_results``. For example, ``total_results_analysis_mean``,
``cluster_results_amb_analysis_mean_write_month`` and
``point_results_waked_all_aggregate_all_mean_write_n10`` select independent
field/filter/group/mean/file policies. All day/week/month/year periods are
available. Filename stems distinguish field/filter/aggregation configurations.
Set ``variables`` on an explicit collector for a custom selection.
``farm_results_all`` and
``point_results_all`` remain unrestricted field selectors, not aggregation.
``total_results_all`` combines every turbine and retains every available field;
use ``total_results_analysis`` to request derived performance metrics automatically.

WindIO ``turbine_outputs`` accepts ``aggregate_turbines`` and boolean
``aggregate_states``. State subsets are applied before reduction and fields
retain WindIO names (``power``, ``rotor_effective_velocity``); grouped axes use
``wind_farm`` or ``cluster``. WindIO ``flow_field`` accepts ``aggregate_states``
only: files retain their spatial grid and reject ``aggregate_points``.
Normalized flow means preserve ``wind_speed``/``wind_direction`` names and
remove ``time``. ``SliceData`` and ``SlicesData`` expose the same explicit flag;
plain ``reduction="mean"`` retains global weighted-sum semantics.

.. code-block:: python

    from foxes.models.outputs import FarmResultsOutput, PointResultsOutput

    farm_summary = FarmResultsOutput(
        variables=[FV.P, FV.EFF, FV.CAPF, FV.YLD],
        aggregate_turbines="wind_farm",
        aggregate_states=True,
    )
    point_summary = PointResultsOutput(
        variables=[FV.WS],
        aggregate_points="all",
        aggregate_states=True,
    )

``Downwind.calc_farm(outputs=...)`` accepts a model-book output name, an
``Output`` instance, or a list of either. It does not accept physical variable
names as output models. A single output with one returned result is unwrapped.
Multiple outputs return a result-name keyed mapping. Writers do not return a
dataset; writer-only or disabled collection returns an empty mapping.

The following example uses only synthetic inflow and exercises the implemented
NumPy farm path:

.. code-block:: python

    import pandas as pd
    import foxes
    import foxes.constants as FC
    import foxes.variables as FV
    from foxes.models.outputs import FarmResultsOutput, FarmResultsCountWriter

    farm = foxes.WindFarm()
    farm.add_turbine(
        foxes.Turbine(xy=[0.0, 0.0], turbine_models=["NREL5MW"]), verbosity=0
    )
    states = foxes.input.states.StatesTable(
        pd.DataFrame({FV.WS: [7.0, 8.0, 9.0], FV.WD: [270.0] * 3}),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
    )
    algo = foxes.algorithms.Downwind(farm, states, wake_models=[], verbosity=0)
    outputs = [
        FarmResultsOutput(variables=[FV.P, FV.AMB_P, FV.WEIGHT]),
        FarmResultsCountWriter(
            base_name="farm",
            n_states_per_file=2,
            out_dir="results",
            variables=[FV.P],
            write_nc_pars={"verbosity": 0},
        ),
    ]
    with foxes.Engine.new("numpy", chunk_size_states=1, verbosity=0):
        results = algo.calc_farm(outputs=outputs)
    assert results[FC.FARM_RESULTS].sizes[FC.STATE] == 3

Engine parity tests cover single, NumPy, threads, process, multiprocess, local
Dask and distributed local clusters. MPI runtime checks are opt-in and have
also been exercised locally; Ray runtime parity remains unverified here.
Iterative and Sequential use explicit intermediate collectors.
Iterative initializes and executes requested outputs only in its final evaluation
pass; convergence iterations use a separate internal collector. Use Iterative
for streamline wake frames. Animations therefore contain final states once,
not repeated intermediate iterations.
Unaggregated collectors preserve supplied weights; normalized state means are
selected explicitly through ``aggregate_states=True``.

Point collection and writers
----------------------------
``PointResultsOutput`` assembles state/target tiles under ``FC.POINT_RESULTS``.
Select fields through ``variables``, ``ambient_vars`` and ``waked_vars``, as
for farm collection. Coordinates remain present for empty field selections.
``calc_points()`` retains the public ``(state, point)`` dataset interface.
The model-book default farm field set is unchanged. For a downstream wake-point
calculation, request ``algo.calc_farm(outputs="farm_results_all")`` to retain
all required wake inputs instead of expanding the default selection.

The collectors live in ``foxes.models.outputs.farm_results`` and
``foxes.models.outputs.point_results``. All farm writers live in
``foxes.models.outputs.farm_results_writers``; all point writers live in
``foxes.models.outputs.point_results_writers``. Public classes are re-exported
from ``foxes.models.outputs``.

* ``PointResultsChunkWriter`` writes each engine-supplied ``TData`` tile on
  its worker to ``<base_name>_<state chunk:06d>_<target chunk:06d>.nc``.
  The engine, not the output, determines both chunk axes.
* ``PointResultsCountWriter`` assembles complete target slabs and writes
  ``<base_name>_<file index:06d>.nc`` with exactly ``n_states_per_file`` states,
  except for the final remainder.
* ``PointResultsPeriodWriter`` assembles complete target slabs and writes
  day/ISO-week/month/year files with the same chronology validation, suffixes
  and empty-period handling as farm writers.

Unaggregated point files preserve internal ``(state, target, target_point)`` coordinates and
fields, including dynamic ``X``, ``Y`` and ``H`` and every target point.
They do not apply the algorithm-return conversion that selects target point
zero and renames ``target`` to ``point``. Writers return no dataset and have
independent field selections/private buffers; compose a collector to obtain
an in-memory dataset too. All writers support ``out_dir``, ``out_fname_fun``,
``write_nc_pars`` and inherited/explicit write verbosity; failures propagate.

The separate point variable-set dictionary defines ``default`` as WS, WD, TI,
RHO, their ambient counterparts and weights, and ``all`` as every available
point field. Model-book selectors mirror farm outputs:

* ``point_results`` / ``point_results_default`` and ``point_results_all``.
* ``point_results_amb``, ``point_results_amb_<variables>`` and
  ``point_results_waked_<variables>``.
* ``point_results_write_n<n_states_per_file>`` (all available fields),
  ``point_results_write_chunks`` and ``point_results_write_day``,
  ``point_results_write_week``, ``point_results_write_month``,
  ``point_results_write_year`` (default fields).

Use ``point_results_all`` instead of the default collector when custom point
models or precomputed datasets supply additional fields. For example:

.. code-block:: python

    import numpy as np
    from foxes.models.outputs import PointResultsCountWriter

    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=2, verbosity=0
    ):
        farm_results = algo.calc_farm(outputs="farm_results_all")
        results = algo.calc_points(
            farm_results,
            np.array([[400.0, 0.0, 90.0], [600.0, 0.0, 90.0]]),
            outputs=[
                "point_results",
                PointResultsCountWriter(
                    "points", 2, variables=[FV.WS], out_dir="results",
                    write_nc_pars={"verbosity": 0},
                ),
            ],
        )
    assert FC.POINT_RESULTS in results

All engines share parent-side progress. ``progress_bar=True`` chooses terminal
or notebook tqdm bars, including Unicode terminal bars where supported, and
newline-only snapshots for redirected logs. ``False`` forces snapshots;
``None`` or verbosity zero disables progress. Output finalization and failures
are distinguished from completed chunk counts.

Flow animations
---------------
``FlowAnimation`` is an optional adopted output, not an engine hook.
It is registered as ``ModelBook.outputs["create_gif"]`` and
``ModelBook.outputs["create_mp4"]``, with base filenames ``ani.gif`` and
``ani.mp4``. Select either with ``algo.calc_farm(outputs="create_mp4")``, for
example. The factories ``"chunk_gif_<chunk>"`` and ``"chunk_mp4_<chunk>"``
select a single state chunk, such as ``outputs="chunk_mp4_1"``.
Its constructor only retains configuration; validation, private selections and
point-model preparation happen in ``initialize()``. Configure a registered
instance before selecting it when non-default grid, filename or directory
settings are needed.
It collects numerical payloads during ``calculate()``, buffers them in
``update_results()``, then joins target tiles and state chunks in engine index
order during ``finalize_results()``. Only finalization renders and saves files.
Each animated variable produces one ``<base_name stem>_<variable>.gif`` or
``.mp4``, without chunk suffixes, even when only one variable is selected.
Set ``base_name="flow.mp4"`` to write MP4 rather than GIF, producing
``flow_WS.mp4`` for wind speed. The final filename suffix,
including any ``out_fname_fun`` transformation, selects the writer: ``.gif``
uses Pillow and ``.mp4`` uses external ffmpeg, which must be installed.
No separate format option is needed. Write failures propagate, and figures
are closed even after failures.
``fps`` accepts any positive finite frame rate, including fractional rates.
Both animation classes announce ``Writing <path>`` when their ``verbosity``
is at least two, followed by ``Finished writing <path>`` after success.
It defaults to ``None``, inheriting ``algo.verbosity``; explicit levels zero
and one suppress both writing messages independently of algorithm progress.
Both animation classes accept ``layout_pars`` for horizontal flow-axis layout
annotations; do not supply ``fig`` or ``ax``.

For farm-only chunks, a static grid in the constructor-selected ``plane`` is loaded once into
``loaded_data["extra_data"]`` as a top-level NumPy array, making it eligible for
engine shared storage. Bounds include all turbine positions, including moving
layouts; the default plane is the midpoint of the minimum/maximum hub heights.
``grid_pars`` accepts ``get_grid_xy`` options such as ``resolution`` or
``n_img_points`` (mutually exclusive), explicit bounds and ``z`` in metres.
Workers broadcast grid coordinates only over their active states and evaluate
blocks of at most the active engine's ``chunk_size_points``; ``None`` evaluates
the whole grid in one block. Point models are initialized before running.
Set ``ambient=True`` on the animation output to evaluate generated grids without
wakes; supplied target data is always reused unchanged.

When ``tdata`` is provided, its coordinates and calculated fields are used
directly, without grid evaluation. All target quadrature locations are retained.
Static rectangular targets use the configured plane's flow-plot renderer; irregular
or moving targets use coloured scatter frames in that plane without interpolation.
``plot_pars`` configures the shared flow-plot renderer for rectangular grids,
or to Matplotlib ``scatter`` for other targets; ``vmin`` and ``vmax`` select a
shared colour scale in either case, and ``vlabel`` overrides the colour-bar
label. Known variables use existing FOXES unit labels. Requested fields, wind
speed and wind direction must be present in supplied target data.

``chunk`` selects all chunks, a state-chunk index, an exact state/target pair,
or a list of either. State labels/positions select from the calculation's input
states, preserving input order. Selecting every input state is represented by
``None``, without a loaded mask. Partial selections are boolean variables with
dimensions ``(FC.STATE,)`` in ``loaded_data["data_vars"]``; engines slice the
mask with each state chunk. Unknown selections fail explicitly; an empty
selection writes no files. Standard ``out_dir`` and ``out_fname_fun`` apply to
every final file.
Selected state chunks must contain the same total number of target points.
No runtime buffers are stored on model instances. Final animation frames are
retained in memory; point-block limits bound scientific evaluation, not the
complete rendered animation.
Raw constructor dictionaries and selection lists move to the normal lifecycle
stash during running and are restored afterwards when a stash is retained.
The running model holds only small configuration, keys and initialized submodels,
not grids, state-sized selections or frame buffers.

For example, continuing the synthetic setup above:

.. code-block:: python

    import numpy as np
    from foxes.models.outputs import FlowAnimation

    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=128, verbosity=0
    ):
        farm_results = algo.calc_farm(outputs=[
            FarmResultsOutput(),
            FlowAnimation(
                base_name="flow.gif",
                out_dir="results",
                grid_pars={"n_img_points": (12, 8)},
            ),
        ])[FC.FARM_RESULTS]
        point_results = algo.calc_points(
            farm_results,
            np.array([[0.0, 0.0, 90.0], [400.0, 0.0, 90.0]]),
            output=FlowAnimation(base_name="targets.gif", out_dir="results"),
        )
    assert point_results.sizes[FC.POINT] == 2

``calc_points(outputs=...)`` still selects physical fields for the returned
dataset. Its separate ``output=...`` parameter adds an output model alongside
the standard collector without changing the dataset return contract.

Flow animations with turbine traces
-----------------------------------
``FlowTurbineAnimation`` derives from ``FlowAnimation`` and adds
a synchronized lower panel rendered by the animation helpers. Set
``turbines`` to original farm indices (all turbines by default) and
``turbine_variable`` to a state-turbine farm field (``FV.REWS`` by default).
Trace data follows the same state/chunk selection as the flow field and remains
in private numerical payloads rather than on the running model.
Both rectangular and arbitrary-target flow rendering use this panel.

``layout_pars`` optionally forwards annotation settings to
the shared layout renderer on the flow axes; do not supply ``fig`` or
``ax``. ``ref_xy`` optionally draws a red cross at two finite coordinates
in metres. The panels have a three-to-one height ratio and use inherited
``figsize``, grid/plot settings, filename suffixes, writers and cleanup.
For example:

.. code-block:: python

    from foxes.models.outputs import FlowTurbineAnimation

    animation = FlowTurbineAnimation(
        turbines=[0],
        turbine_variable=FV.REWS,
        base_name="flow.gif",
        grid_pars={"n_img_points": (12, 8)},
        plot_pars={"vmin": 0, "rotor_color": "red"},
        layout_pars={"annotate": 1, "alpha": 0, "title": ""},
        ref_xy=[300.0, 200.0],
    )
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        algo.calc_farm(outputs=animation)

The model book registers this class as ``create_turbine_gif`` and
``create_turbine_mp4``, with base filenames ``ani.gif`` and ``ani.mp4``.
They trace all turbines' REWS by default and write ``ani_WS.gif`` or
``ani_WS.mp4``. These filenames match the flow-only outputs; configure
different base names before selecting both variants of the same format,
or the later output overwrites the earlier file.
Select these names directly with ``calc_farm(outputs=...)`` or example ``-o``.
The ``create_gif`` and ``create_mp4`` registrations remain ``FlowAnimation``.
The ``dyn_wakes`` example configures its own ``create_turbine_gif`` and
``create_turbine_mp4`` registrations to reproduce its flow-and-REWS animation.

Sequential animation outputs
----------------------------
Animation composition lives in ``foxes.models.outputs.Animator``.
Sequential plugins live in ``foxes.models.seq_plugins``:
``SeqAnimationOutputPlugin``, ``SeqWakeDebugPlugin`` and ``SeqFlowAnimationPlugin``.
These helpers are not standalone chunk-output models.
Their former ``foxes.output`` exports and modules are removed.

``foxes.models.seq_plugins.SeqAnimationOutputPlugin`` adapts selected ``FlowAnimation``
models (including ``FlowTurbineAnimation``) to the Sequential iterator.
Pass model-book output names or model instances to the plugin, not to
``Sequential(outputs=...)``: that argument still selects physical variables.
Other output model types are rejected.

.. code-block:: python

    algo = foxes.algorithms.Sequential(
        farm, states, wake_models=["Jensen_linear_k007"],
        wake_frame="seq_dyn_wakes_1min",
        plugins=[foxes.models.seq_plugins.SeqAnimationOutputPlugin(["create_gif"])],
    )
    algo.mbook.outputs["create_gif"].grid_pars = {"n_img_points": (12, 8)}
    with foxes.Engine.new("single", chunk_size_points=128, verbosity=0):
        for farm_state in algo:
            pass

Point fields are calculated during each selected state step, while wake history
is available, respecting the engine's point chunk limit. The plugin owns
numerical buffers outside the output model; rendering happens once when
iteration completes. Abandoned iteration does not write an animation.
State selections refer to full input states; chunk indices refer to sequential
step indices with target index zero. ``debug=True`` adds the existing wake
point/vector overlays without changing their scientific calculations.

Tables, maps, arbitrary points and metadata plots
--------------------------------------------------
All concrete outputs belong to ``foxes.models.outputs`` and follow the same
constructor-defined lifecycle. No separate dataset-bound output package remains.

``StateTurbineTable(variables=[...], name_map={...}, isel={...}, sel={...},
transpose=True, to_file="table.nc")`` collects only selected fields. Dimension
selection uses names after renaming; transposition places turbine before state.
The optional NetCDF file is written once after parent-side assembly.
``StateTurbineMap(variable=FV.P, plot_pars={...})`` renders the selected table;
``plot=False`` or ``nofig=True`` returns its numerical dataset. Singleton,
datetime and string state coordinates are supported.

``PointCalculator(points=[...], variables=[FV.WS], states_mean=True)`` evaluates
arbitrary static or state-dependent metre coordinates with normal point chunks.
Coordinates are loaded before execution and raw configuration is stashed while
running. Supplied point results take precedence. State means transport only
weighted sums and mass, normalize on the parent and ignore zero-weight NaNs.
Returned datasets include ``x``, ``y``, ``z`` and configured fields; raw fields
have ``(state, point)`` dimensions and means have ``(point,)``.

``RotorPointPlot`` reads the executing algorithm's initialized rotor design,
without farm-state payloads. ``TurbineTypeCurves`` consumes an explicitly
configured no-wake wind-speed scan: each turbine is one comparison curve.
Set scan states, turbine types and power-mask models on the algorithm; the
output never launches hidden simulations or mutates a model book.

YAML uses these same constructors and stored ``$name`` references. The small
``output_type: plt`` exception dispatches its ``functions`` directly to
``matplotlib.pyplot``, outside the output lifecycle. Stored Matplotlib figures
and axes support ``object: $name`` method calls for composition and saving.
``nofig`` skips all these calls, including display and file writing.
``savefig`` paths resolve relative to the configured output directory.
There is no pyplot output class or model-book entry.

``SliceData`` and ``FlowPlots2D`` share static grid coordinates. When coordinates
are returned alongside data, the grid array has shape ``(n_points, 3)``.
XY, XZ, and YZ grids share
those coordinates across all selected states. Point calculations use the active
engine to broadcast coordinates over each state chunk; callers do not need to
repeat the grid themselves.

Mean slices reduce state-by-point fields inside each worker and transport only
weighted point sums. State-resolved slice results necessarily retain their
selected state frames.

.. toctree::
    :maxdepth: 2

    _autoapi/foxes/core/output/index
    _autoapi/foxes/models/outputs/index
    _autoapi/foxes/models/seq_plugins/index
