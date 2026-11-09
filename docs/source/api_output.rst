Output models
=============
Adopted outputs derive from ``foxes.core.Output`` and are exported from
``foxes.models.outputs``. Engine workers calculate chunk-local NumPy payloads;
the output's ``update_results()`` and ``finalize_results()`` hooks unify them.
``OutputList`` composes models in declaration order and merges their required
input variables.

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
``normalize=True`` normalizes collected farm weights after finalization and
requires a collected weighted dataset; it does not rewrite output files.

Point collection
----------------
``PointResultsOutput`` assembles state/target tiles under ``FC.POINT_RESULTS``.
``calc_points()`` retains the public ``(state, point)`` dataset interface.
The model-book default farm field set is unchanged. For a downstream wake-point
calculation, request ``algo.calc_farm(outputs="farm_results_all")`` to retain
all required wake inputs instead of expanding the default selection.

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
is positive. It defaults to ``None``, inheriting ``algo.verbosity``; an explicit
zero suppresses the message independently of algorithm progress.
Both animation classes accept ``layout_pars`` for flow-axis annotations through
``FarmLayoutOutput.get_figure``; do not supply ``fig`` or ``ax``.

For farm-only chunks, a static XY grid is loaded once into
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
Static rectangular XY targets use the existing flow-plot renderer; irregular
or moving targets use coloured XY scatter frames without interpolation.
``plot_pars`` passes to ``FlowPlots2D.gen_states_fig_xy`` for rectangular grids,
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
a synchronized lower panel using ``FarmResultsEval.gen_stdata``. Set
``turbines`` to original farm indices (all turbines by default) and
``turbine_variable`` to a state-turbine farm field (``FV.REWS`` by default).
Trace data follows the same state/chunk selection as the flow field and remains
in private numerical payloads rather than on the running model.
Both rectangular and arbitrary-target flow rendering use this panel.

``layout_pars`` optionally forwards annotation settings to
``FarmLayoutOutput.get_figure`` on the flow axes; do not supply ``fig`` or
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
``foxes.output.SeqAnimationOutputPlugin`` adapts selected ``FlowAnimation``
models (including ``FlowTurbineAnimation``) to the Sequential iterator.
Pass model-book output names or model instances to the plugin, not to
``Sequential(outputs=...)``: that argument still selects physical variables.
Other output model types are rejected.

.. code-block:: python

    algo = foxes.algorithms.Sequential(
        farm, states, wake_models=["Jensen_linear_k007"],
        wake_frame="seq_dyn_wakes_1min",
        plugins=[foxes.output.SeqAnimationOutputPlugin(["create_gif"])],
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

Existing consumers awaiting adoption
------------------------------------
The remaining ``foxes.output`` classes have not been adopted into the new
chunk-output structure. They are not newly supported chunk models; their
existing import paths use the core base directly. Evaluation methods
with names such as ``calculate`` or ``get_data`` remain consumer APIs, not
chunk hooks; passing such consumers to ``calc_farm(outputs=...)`` is unsupported.

``SliceData`` and ``FlowPlots2D`` return grid-data tuples whose final item is a
static coordinate array with shape ``(n_points, 3)``. XY, XZ, and YZ grids share
those coordinates across all selected states. Point calculations use the active
engine to broadcast coordinates over each state chunk; callers do not need to
repeat the grid themselves.

Mean slices still collect state-by-point results before applying their weights.
Static coordinate storage removes grid replication, but does not make that
result collection memory-bounded.

.. toctree::
    :maxdepth: 2

    _autoapi/foxes/core/output/index
    _autoapi/foxes/models/outputs/index
    _autoapi/foxes/output/index
    _autoapi/foxes/output/flow_plots_2d/index
    _autoapi/foxes/output/seq_plugins/index
