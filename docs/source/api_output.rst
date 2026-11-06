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

The NumPy and thread farm paths are covered by regression tests. Other engines,
Iterative, Sequential and point-output integration still require migration.
``normalize=True`` normalizes collected farm weights after finalization and
requires a collected weighted dataset; it does not rewrite output files.

Existing consumers awaiting adoption
------------------------------------
The remaining ``foxes.output`` classes have not been adopted into the new
chunk-output structure. They are not newly supported chunk models, and this
branch's existing import paths still require migration work.

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
