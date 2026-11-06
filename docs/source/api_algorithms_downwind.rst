Downwind algorithm
==================
Orders turbines in downwind direction in a single sweep.

Chunk-based farm outputs
------------------------
``calc_farm()`` selects output models through ``outputs``. The default
``"farm_results"`` model-book entry returns an xarray farm dataset. Pass an
``Output`` instance or a model-book output name for one output, or a list for
an ordered output chain. Physical field selections belong to the output model:

.. code-block:: python

    from foxes.models.outputs import FarmResultsOutput
    import foxes.variables as FV

    selected_output = FarmResultsOutput(variables=[FV.P, FV.AMB_P])
    # Pass selected_output as algo.calc_farm(outputs=selected_output).

Multiple outputs return a result-name keyed mapping, even when only one output
returns a dataset. Writer-only outputs and disabled collection (``None`` or an
empty list) return an empty mapping. Use explicit NumPy or thread engines for
the currently validated farm-output path; the other engine/algorithm and
point-output paths still require migration.

See :doc:`api_output` for dataset and NetCDF output models, composition and a
runnable synthetic example.

Ambient-only calculations
-------------------------
Use ``ambient=True`` to skip wake effects. Farm result filtering is independent
of this calculation choice: ``FarmResultsOutput(ambient_vars=True,
waked_vars=False)`` collects ambient fields while omitting ordinary fields
that have ambient counterparts. Choose both flags to retain both selected
field names. Farm ``outputs`` are output models, not lists of variable names.

``calc_points()`` retains its ``ambient_keep=False`` parameter and
variable-name output selection, but its engine integration is not yet adopted
into the chunk-output structure. Do not infer point or Iterative support from
the farm-output tests.

.. toctree::
    :maxdepth: 2

    _autoapi/foxes/algorithms/downwind/index
    _autoapi/foxes/algorithms/downwind/models/index
