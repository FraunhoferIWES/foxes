Downwind algorithm
==================
Orders turbines in downwind direction in a single sweep.

Ambient-only calculations
-------------------------
Use ``ambient=True`` to skip wake effects. By default, results omit ordinary
variables that have ambient counterparts. Set ``ambient_keep=True`` to retain
both names, such as ``P`` and ``AMB_P`` or ``REWS`` and ``AMB_REWS``; each retained
pair contains the same unwaked values. This option applies to both
``calc_farm()`` and ``calc_points()`` and defaults to ``False``.

For example, using synthetic uniform inflow:

.. code-block:: python

    import foxes
    import foxes.variables as FV

    farm = foxes.WindFarm()
    farm.add_turbine(
        foxes.Turbine(xy=[0.0, 0.0], turbine_models=["NREL5MW"]), verbosity=0
    )
    states = foxes.input.states.SingleStateStates(
        ws=8.0, wd=270.0, ti=0.08, rho=1.225
    )
    algo = foxes.algorithms.Downwind(farm, states, wake_models=[], verbosity=0)
    with foxes.Engine.new("single", verbosity=0):
        farm_results = algo.calc_farm(ambient=True, ambient_keep=True)
        point_results = algo.calc_points(
            farm_results, [[1000.0, 0.0, 90.0]], ambient=True, ambient_keep=True
        )
    assert FV.P in farm_results and FV.AMB_P in farm_results
    assert FV.WS in point_results and FV.AMB_WS in point_results

Explicit ``outputs`` selections still select exactly the requested variables;
request both names to return both. ``ambient_keep`` has no effect on ordinary
waked calculations. Iterative inherits the same option and retains ordinary
fields during intermediate farm passes regardless of the final output setting.

.. toctree::
    :maxdepth: 2

    _autoapi/foxes/algorithms/downwind/index
    _autoapi/foxes/algorithms/downwind/models/index
