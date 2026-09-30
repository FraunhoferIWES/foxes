foxes.output
============
Classes and functions that create output from *foxes* calculation results.
``FarmResultsEval.calc_farm_capacity_factor`` returns the state-weighted mean
farm power divided by total installed farm capacity.

Farm layout rendering
---------------------
``FarmLayoutOutput.get_figure`` uses scatter markers by default. Set
``true_turbine_radii=True`` to render filled circles at the physical rotor
radii in planar data coordinates. Direct colors and ``color_by`` values use
the same fills, colormaps, limits, and opacity as scatter markers. Physical
radii require finite positive turbine diameters and are unavailable for
longitude/latitude plots.

.. toctree::
    :maxdepth: 2

    _autoapi/foxes/output/index
    _autoapi/foxes/output/flow_plots_2d/index
    _autoapi/foxes/output/seq_plugins/index
