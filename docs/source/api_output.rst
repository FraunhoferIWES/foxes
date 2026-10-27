foxes.output
============
Classes and functions that create output from *foxes* calculation results.

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

    _autoapi/foxes/output/index
    _autoapi/foxes/output/flow_plots_2d/index
    _autoapi/foxes/output/seq_plugins/index
