.. _models:

Models
======

Model types
-----------

The results of *foxes* runs depend on a number of model choices by the user:

* :ref:`Rotor models <rotor-models>`: Evaluate the flow field at the rotor and compute ambient rotor equivalent quantities.
* :ref:`Turbine types <turbine-types>`: Define rotor diameter and hub height, and provide thrust coefficient and power yield depending on rotor equivalent quantities.
* :ref:`Wake models <wake-models>`: Compute wake deltas for flow quantities in the wake of a turbine.
* :ref:`Wake frames <wake-frames>`: Determine the path of wake propagation and the local coordinate system that follows the centreline of the wake.
* :ref:`Wake deflections <wake-deflections>`: Bend the wakes for rotors with yaw misalignment. This effect is added to the wake frame path.
* :ref:`Partial wakes <partial-wakes>`: Compute rotor disc averages of wake effects, i.e., the partial wakes models calculate the rotor effective wake deltas.
* :ref:`Turbine models <turbine-models>`: Each wind turbine within the wind farm can have individual turbine model choices. For each state and turbine, those compute data from currently existing data.
* :ref:`Farm controllers <farm-controllers>`: Responsible for running the turbine models, in a certain order and optionally depending on conditions.
* :ref:`Ground models <ground-models>`: Add ground effects to the wake calculation, for example the reflection from horizontal planes.
* :ref:`Point models <point-models>`: Calculate point-based data during ``algo.calc_points()`` or modify ambient-state results.
* :ref:`Vertical profiles <vertical-profiles>`: Analytical vertical profiles transform uniform ambient states into height dependent inflow.
* :doc:`Output models <outputs>`: Return result datasets, write files and create figures from farm or point calculations.

All concrete models are stored in the so-called :code:`ModelBook` object under
a name string (or a name string template that is provided by a so-called model factory),
see the model book example.

.. _rotor-models:

Rotor models
------------
Rotor models evaluate the flow field at the rotor and compute ambient rotor equivalent quantities, for example the rotor averaged
background wind speed. The list of available models and also rotor point visualizations can be found in the rotor model visualization example.

.. _turbine-types:

Turbine types
-------------
Turbine type models define rotor diameter and hub height, and provide thrust coefficient and power yield depending on rotor equivalent quantities.
They are either chosen from the provided static data by their model name (e.g. *DTU10MW*),
or added to the model book.

For example, a turbine type of a 5 MW turbine based on a csv file with
columns *ws* for wind speed, *P* for power and *ct* for thrust
coefficients can be added as

    .. code-block:: python

        mbook = foxes.ModelBook()
        mbook.turbine_types["turbine1"] = foxes.models.turbine_types.PCtFile(
            "turbine1.csv",
            col_ws="ws",
            col_P="P",
            col_ct="ct",
            D=100.5,
            H=120,
            P_nominal=5000,
            P_unit="kW",
        )

If the file name follows the convention

:code:`name-<power>MW-D<rotor diameter>-H<hub height>.csv`

where `d` replaces the dot for digits, then the above could be reduced to

    .. code-block:: python

        mbook.turbine_types["turbine1"] = foxes.models.turbine_types.PCtFile(
            "turbine1-5MW-D100d5-H120.csv",
            col_ws="ws",
            col_P="P",
            col_ct="ct",
        )

Turbine type models can also be based on other input data, e.g. `cp` instead
of power, or other input files. The list of available turbine type classes
can be found in the :doc:`turbine types API <_autoapi/foxes/models/turbine_types/index>`.

.. _wake-models:

Wake models
-----------
Wake models compute wake deltas for flow quantities in the wake. Wind speed deficits and turbulence
intensity deltas are often computed by two separate wake models, but could also stem from a single model.

The wake model classes can be found in the :doc:`wake models API <_autoapi/foxes/models/wake_models/index>`.
They are organized into three sub-packages, according to their purpose and target variables:

* :doc:`wind <_autoapi/foxes/models/wake_models/wind/index>`: Wind deficit models, computing negative deltas for the wind speed variable `WS`,
* :doc:`ti <_autoapi/foxes/models/wake_models/ti/index>`: Positive wake deltas acting on the variable `TI`, modelling the turbulence increase within the wake region,
* :doc:`induction <_autoapi/foxes/models/wake_models/induction/index>`: Individual turbine induction models acting as wind speed deltas, which, in combination, model wind farm blockage effects.

Note that `wind` and `ti` wake models affect downstream turbines, while `induction` models
mainly affect upstream and stream-orthogonal turbines. During calulations, a list of
wake models is expected, so in principle, a wind deficit model, a TI wake model and a turbine
induction model can be combined. If an induction model is included in the
list of model selections, the :doc:`Iterative algorithm <api_algorithms_iterative>` has to be applied.

All wake model classes are implemented according to their mathematical nature, i.e.,
if applicable, they are derived from one of the following types:

* :class:`DistSlicedWakeModel<foxes.models.wake_models.dist_sliced.DistSlicedWakeModel>`: The wake delta depends on the wake frame coordinate `x` differently than on `(y, z)`, e.g., the `x` dependency can be factorized.
* :class:`AxisymmetricWakeModel<foxes.models.wake_models.axisymmetric.AxisymmetricWakeModel>`: Dist-sliced wake with axial symmetry, i.e., the wake can be described by `x` and a radial wake frame coordinate `r`.
* :class:`GaussianWakeModel<foxes.models.wake_models.gaussian.GaussianWakeModel>`: Axisymmetric wake that follows a Gaussian function, where the standard deviation `sigma(x)` depends on `x` only.
* :class:`TopHatWakeModel<foxes.models.wake_models.top_hat.TopHatWakeModel>`: Axisymmetric wake that is independent of `r` within the top-hat shape, and zero outside.

The reasoning behind this is that the partial wakes models can then
build upon the underlying shape of the wake.

Wake superposition is part of the responsibility of the wake model. Most models expect
a choice of the underlying :doc:`wake superposition model <_autoapi/foxes/models/wake_superpositions/index>`
in their constructor, in terms of their respective name in the model book.
Examples are `ws_linear` for linear wind deficit superposition, or `ti_quadratic`
for quadratic TI wake increase superposition.

For wind-speed superpositions, the base name scales deficits by the source turbine's
waked wind speed, while the `_amb` suffix selects the source ambient wind speed. The
`_target` suffix selects the target turbine's waked wind speed, and `_amb_target`
selects its ambient wind speed. The same choices are available through the
`scale_amb` and `scale_target` constructor parameters.

The vector superpositions ``vector``, ``vector_amb``, ``vector_target``,
``vector_amb_target`` and ``vector_loc`` support the same scaling choices.
They do not support the ``_lim`` variants of the scalar wind-speed superpositions.

`ws_product` is different: it combines dimensionless deficits and applies the
result to the target ambient wind speed, so a separate `_amb_target` variant
would be identical.

The list of wake model name templates in the default model book is long,
but that is mainly due to variations of various constructor argument choices. Typical examples are

* `Jensen_<superposition>_[wake_k]`: The classic Jensen wind deficit model,
* `Bastankhah2014_<superposition>_[wake_k]`: The Gaussian wind deficit model by `Bastankhah and Porté-Agel from 2014 <https://doi.org/10.1016/j.renene.2014.01.002>`_,
* `Niayifar`: A Bastankhah2014 model-book preset using the local-TI wake-growth relation from `Niayifar and Porté-Agel <https://doi.org/10.3390/en9090741>`_,
* `Bastankhah2016_<superposition>_[wake_k]`: The wind deficit model by `Bastankhah and Porté-Agel from 2016 <https://doi.org/10.1017/jfm.2016.595>`_,
* `TurbOPark_<superposition>_[wake_k]`: The Gaussian wind deficit model by `Pedersen et al. from 2022 <https://iopscience.iop.org/article/10.1088/1742-6596/2265/2/022063/pdf>`_,
* `CrespoHernandez_<superposition>_[wake_k]`: The top-hat TI addition wake model by `Crespo and Hernandez from 1996 <https://doi.org/10.1016/0167-6105(95)00033-X>`_,
* `IECTI2019_<superposition>`: The top-hat TI addition wake model by `Frandsen from 2019 <https://backend.orbit.dtu.dk/ws/files/3750291/2009_31.pdf>`_.

Note that in all above cases, the `superposition` parameter is
any of the available :doc:`wake superposition models <_autoapi/foxes/models/wake_superpositions/index>` for wind speed or TI, depending on the model.
Here the convention is that you write `linear` for the choice `ws_linear` or `ti_linear`, etc., depending if the wake model targets wind speed or TI
(cf. the model book example).

The `[wake_k]` part of the model name can be replaced by one of the following patterns:

* `k<k>`, where `<k>` is to be replaced by the value for the wake growth factor `k`, with dot-skipping convention (e.g. `004` for the value `0.04`, etc.)
* ``k0_<k0>`` for a constant intercept, or ``k1_<k1>`` for a TI multiplier with zero intercept.
* ``k0_<k0>_k1_<k1>`` for ``k = k_0 + k_1 * TI``.
* ``ambk0_<ambk0>``, ``ambk1_<ambk1>`` or
  ``ambk0_<ambk0>_ambk1_<ambk1>`` for the corresponding relation using ``AMB_TI``.
* nothing, e.g. `Bastankhah2014_linear`, which searches the value for `k` in the list of available farm data. This is intended to be used whenever a turbine model computes the `k` values, typically the the :class:`kTI<foxes.models.turbine_models.kTI_model.kTI>` turbine model, or an optimizer.

Python wake constructors accept ``k_0`` and ``k_1``. The canonical coefficient
fields are ``FV.K_0 = "k0"`` and ``FV.K_1 = "k1"``; both are dimensionless.
An omitted coefficient is zero when the other is supplied. A zero multiplier
does not require TI. Explicit ``k`` and linear coefficients are mutually exclusive.
The underscore between a coefficient label and its value is required:
``k0_004`` selects an intercept of 0.04, while the existing ``k004`` constant
growth notation is unchanged.

Removed FOXES keywords ``ka``, ``kb``, ``kti`` and ``kTI`` and old parameterized
model-book names raise ``DeprecationWarning`` as an exception, with migration
guidance. Replace the former intercept ``kb`` with ``k_0`` and the former
multiplier ``ka``/``kti``/``kTI`` with ``k_1``.

WindIO schema keys remain unchanged. Following WIFA's
``k = k_a * TI + k_b``, ``wake_expansion_coefficient.k_a`` maps to
``k_1``, ``k_b`` maps to ``k_0``, and ``free_stream_ti`` selects ambient TI.
This translation applies to wind deficits and turbulence models, including
inherited coefficients. This corrects FOXES's previous reversed interpretation.
Existing configurations written for that interpretation must swap the two
coefficient values to preserve their results; for example, constant growth
of 0.04 is now ``k_a: 0.0, k_b: 0.04``.

Coefficient tokens also accept standard floating-point notation, for example
``Jensen_linear_k0.04`` and ``Jensen_linear_k004`` select the same growth factor.
The integrated model ``TurbOParkIX_<superposition>_k<k>_dx<dx>`` accepts a
constant growth factor, while ``TurbOParkIX_<superposition>_dx<dx>`` reads it
from farm data. Its factories do not offer linear-coefficient suffixes;
the Python constructor rejects a nonzero ``k_1``.

Examples for valid wake model choices are:

* `Jensen_quadratic_k0075`
* ``Bastankhah2014_linear_k0_0012_k1_02``
* ``Bastankhah2016_linear_lim_ambk1_04``
* `TurbOPark_quadratic_amb_target_k004`
* ``CrespoHernandez_max_k0_003_k1_0213``
* `Bastankhah2014_linear`
* `Niayifar`

.. _wake-frames:

Wake frames
-----------
Wake frames determine the path of wake propagation, for example parallel to the
wind direction at the rotor, or along a streamline, and the local coordinate system
that follows the centreline of the wake.

Wake frames also determine the downwind
order of the turbines, so chosing straight wakes for cases with spatially
heterogeneous background flow can cause wrong results in multiple ways.

The wake coordinates are defined as follows:

* The origin is at the rotor centre,
* the `x` coordinate folows the centreline path of the wake,
* the `z` coordinate starts pointing upwards at the rotor, then follows the centreline orthogonally,
* the `y` coordinate closes the right-handed coordinate frame, i.e., it follows from the cross product of `z` with `x`.

The available wake frame classes are listed
:doc:`here in the API <_autoapi/foxes/models/wake_frames/index>`. The default model book
contains many pre-defined wake frames, for example:

* `rotor_wd`: Straight wakes, following the wind direction measured at the centre of the wake causing rotor.
* `streamlines_<step>`: Streamline (or streaklines) following steady-state wakes, for a virtual time step of `step` seconds. See the heterogeneous-flow example.
* `dyn_wakes`, `dyn_wakes_<length>`: Dynamic flow following wakes for inhomogeneous wind data, optionally with maximal wake length `length`, e.g. `length=8km` or `length=4321m`, or other values with one of those two units. See the Dynamic Wakes 1 example.
* `timelines`, `timelines_<dt>`: Dynamic flow following wakes for spatially homogeneous wind data, optionally with time step `dt`, e.g. `dt=10s` or `dt=1min`, or other values with one of those two units. See the Dynamic Wakes 2 example.
* `seq_dyn_wakes`, `seq_dyn_wakes_<dt>`: Sequential state evaluation (caution: slow, no state chunking), optionally with time step `dt`, e.g. `dt=10s` or `dt=1min`, or other values with one of those two units. See the Dynamic Wakes 3 example.

.. _wake-deflections:

Wake deflections
----------------
For rotors with yaw misalignment the wake is bent and follows a curved path. This is modelled by `WakeDeflection` models, which

* modify the wake path from the wake frame,
* optionally rotate the waked wind vector along the path,
* or, alternatively (and also optionally), they modify the wind deficit along the curved path.

Currently, these are the implemented wake deflection model classes:

* :class:`NoDeflection<foxes.models.wake_deflections.no_deflection.NoDeflection>`: Ignores wake deflection effects (default). Should be used if no yaw misalignment is present.
* :class:`Bastankhah2016Deflection<foxes.models.wake_deflections.bastankhah2016.Bastankhah2016Deflection>`: Extracted from the wake model paper by `Bastankhah and Porté-Agel from 2016 <https://doi.org/10.1017/jfm.2016.595>`_, recommended in combination with the corresponding wake model. Only the wake path is modified by this model.
* :class:`JimenezDeflection<foxes.models.wake_deflections.jimenez.JimenezDeflection>`: Bends any wake according to `Jimenez et al. <https://onlinelibrary.wiley.com/doi/epdf/10.1002/we.380>`_, and optionally also rotates the wind vector within the wake.

The `JimenezDeflection` model has three pre-configured versions in the model book:

* `Jimenez`: Modifies the wake path and rotates the waked horizontal wind vector along the path.
* `JimenezProj`: Modifies the wake path and alters the wind deficit within the wake by projection, but does not rotate the vector.
* `JimenezPath`: Modifies the wake path but does not modifiy the wind vector delta within the wake.

All versions allow for non-standard values of the model parameter `beta`, for example by `Jimenez_b012` for `beta = 0.12` instead of the default `beta = 0.1`.

The `Bastankhah2016Deflection` is called `Bastankhah2016` in the model book, and the `NoDeflection` model `no_deflection`.

For an example in the context on dynamic wakes, see the dynamic wake deflection example.

.. _partial-wakes:

Partial wakes
-------------
Partial wakes models compute rotor disc averages of wake effects, i.e.,
the partial wakes models calculate the rotor effective wake deltas.

Some of the partial wakes models make use of the mathematical structure of
the associated wake model:

* :class:`PartialCentre<foxes.models.partial_wakes.centre.PartialCentre>`: Only evaluate wakes at rotor centres. This is fast, but not accurate.
* :class:`RotorPoints<foxes.models.partial_wakes.rotor_points.RotorPoints>`: Evaluate the wake model at exactly the rotor points, then take the average of the combined result. For large number of rotor points this is accurate, but potentially slow.
* :class:`PartialTopHat<foxes.models.partial_wakes.top_hat.PartialTopHat>`: Compute the overlap of the wake circle with the rotor disc. This is mathematically exact and fast, but limited to wakes with top-hat shapes.
* :class:`PartialAxiwake<foxes.models.partial_wakes.axiwake.PartialAxiwake>`: Compute the numerical integral of axi-symmetric wakes with the rotor disc. This needs less evaluation points than grid-type wake averaging.
* :class:`PartialGaussian<foxes.models.partial_wakes.gaussian.PartialGaussian>`: Gaussian-only partial wakes model using the exact analytical rotor-disc average.
* :class:`PartialGaussianLookup<foxes.models.partial_wakes.gaussian.PartialGaussianLookup>`: Gaussian-only partial wakes model using a precomputed lookup artifact for rotor-effective wake weighting at one rotor-centre target point.
* :class:`PartialSegregated<foxes.models.partial_wakes.segregated.PartialSegregated>`: Abstract base class for segregated wake averaging, which means adding the averaged wake to the averaged background result (in contrast to `RotorPoints`).
* :class:`PartialGrid<foxes.models.partial_wakes.grid.PartialGrid>`: Segregated partial wakes evaluated at points of a :class:`grid-type rotor<foxes.models.rotor_models.grid.GridRotor>` (which is usually not equal to the selected rotor model).

In the default model book, concrete instances of the above partial wakes models
can be found under the names

* `centre`: The centre point model,
* `rotor_points`: The rotor points model,
* `top_hat`: The top-hat model,
* `gaussian`: Analytical Gaussian rotor-disc averaging,
* `gaussian_lookup`: Gaussian lookup-table partial wakes model,
* `axiwake<n>`: The axiwake model, with `n` representing the number of steps for the discretization of the integral over each downstream rotor,
* `grid<n2>`: The grid model with `n2` representing the number of points in a regular square grid.

Partial wakes are selected when constructing the algorithm. Choices can be
specified for the selected wake models:

* by a dictionary, which maps wake model names to model choices (or default choices, if not found),
* or by a list, where the mapping to the wake models is in order of appearance,
* or by a string, in which case all models are either mapped to the given model, or, if that fails with `TypeError`, to their defaults,
* or by `None`, which means all models are mapped to the default choice.

A comparison is available in the
:doc:`partial-wake notebook <notebooks/partial_wakes>`.

Generate a custom Gaussian lookup artifact with:

.. code-block:: console

    foxes_create_gaussian_lookup gaussian_lookup.nc

Use ``--help`` to inspect generation options. See
:class:`PartialGaussianLookup<foxes.models.partial_wakes.gaussian.PartialGaussianLookup>`
for artifact loading, validation, and out-of-bounds policies.

.. _turbine-models:

Turbine models
--------------
Each wind turbine within the wind farm can have individual turbine model choices.
For each state and turbine, those compute data from currently existing data.

The list of available turbine model classes can be found
:doc:`here in the API <_autoapi/foxes/models/turbine_models/index>`. For example:

* :class:`kTI<foxes.models.turbine_models.kTI_model.kTI>`: Computes
  ``FV.K = FV.K_0 + FV.K_1 * TI`` for each state and turbine.
  Supply uniform Python parameters ``k_0``/``k_1`` or farm fields ``k0``/``k1``.
  A missing intercept defaults to zero; a missing multiplier is an error.
  Model-book examples are ``kTI_k1_02``, ``kTI_k0_001_k1_02`` and
  ``kTI_ambk0_001_ambk1_02``. The ``kTI`` and ``kTI_amb`` presets read
  coefficients from farm data, using local and ambient TI respectively.
  Wake models without explicit growth configuration use the computed ``FV.K``.
* :class:`SetFarmVars<foxes.models.turbine_models.set_farm_vars.SetFarmVars>`: Set any farm variable to any state-turbine data array, or sub-array (nan values are ignored), either initially (`pre_rotor=True`) or after the wake calculation.
* :class:`PowerMask<foxes.models.turbine_models.power_mask.PowerMask>`: Curtail or boost the turbine by re-setting the maximal power of the turbine, see the Power mask example.
* :class:`SectorManagement<foxes.models.turbine_models.sector_management.SectorManagement>`: Modify farm variables if wind speed and/or wind direction values are within certain ranges, see the Wind sector management example.
* :class:`YAW2YAWM<foxes.models.turbine_models.yaw2yawm.YAW2YAWM>` and :class:`YAWM2YAW<foxes.models.turbine_models.yawm2yaw.YAWM2YAW>`: Compute absolute yaw angles from yaw misalignment, and vice-versa.
* :class:`Calculator<foxes.models.turbine_models.calculator.Calculator>`: Apply any user-written function that calculates values of farm variables.
* :class:`LookupTable<foxes.models.turbine_models.lookup_table.LookupTable>`: Use a lookup-table for the computation of farm variables.
* :class:`TurbineEvaluation<foxes.models.turbine_models.evaluation.TurbineEvaluation>`: Compute nominal capacity, efficiency, capacity factor, annualized yield and P75/P90 yield at every state and turbine.

Request turbine metrics through output variable selection.
``PrepareForOutput`` applies ``TurbineEvaluation`` automatically after the
ambient/wake power calculations and before restoring original turbine order.
No manual turbine-model registration is needed:

.. code-block:: python

    import foxes
    import foxes.variables as FV
    from foxes.models.outputs import FarmResultsOutput

    farm = foxes.WindFarm()
    farm.add_turbine(
        foxes.Turbine(
            xy=[0.0, 0.0],
            turbine_models=["NREL5MW"],
        ),
        verbosity=0,
    )
    states = foxes.input.states.SingleStateStates(
        ws=8.0, wd=270.0, ti=0.08, rho=1.225,
    )
    algo = foxes.algorithms.Downwind(farm, states, wake_models=[], verbosity=0)
    farm_results = algo.calc_farm(outputs=FarmResultsOutput(
        variables=[FV.CAP, FV.EFF, FV.CAPF, FV.YLD, FV.AMB_YLD],
    ))

The ``farm_results_analysis`` model-book collector selects capacity, capacity
factors, efficiency and weights. Unrestricted collectors such as
``farm_results_all`` retain already available fields; they do not request new
metrics. The default farm-output field selection is unchanged.

Every result retains dimensions ``(state, turbine)`` in ``fdata``. Capacity is
the turbine type's nominal power in its configured power unit. Capacity factor
is ``P / CAP``.
Efficiency is the unclipped ratio ``P / AMB_P``. A zero numerator gives zero,
including ``0 / 0``; a nonzero numerator over either signed zero gives positive
``np.inf`` without division warnings. NaNs propagate otherwise, including an
unknown numerator over zero. Nominal power must still be finite and positive
for capacity-based metrics.

Yield is ``P * P_unit_W * 8760 / 1e9`` in GWh/a, using each turbine type's
power-unit conversion. These are annualized per-state values: apply state
weights only during subsequent aggregation, not inside this model.
``P75 = YLD * (1 - 0.675 * power_uncert)`` and
``P90 = YLD * (1 - 1.282 * power_uncert)`` use the existing normal-uncertainty
approximation. ``power_uncert`` is required for these metrics, finite and
nonnegative; ``0.1`` means 10 percent. These are not empirical percentiles over
states, and summing them assumes the same relative-uncertainty model.
Automatic evaluation rejects probability-yield requests when uncertainty is
unavailable rather than assuming a value.

Ambient variants use ``AMB_P``: ``AMB_CAPF``, ``AMB_YLD``,
``AMB_P75`` and ``AMB_P90``. Request them independently or alongside waked
metrics. Output preparation evaluates final powers, including unwaked turbines,
ambient-only calculations and Iterative's final under-relaxed powers.
State slices, boolean masks and controller turbine selections update only
selected entries; returned arrays always retain the full state-turbine shape.

``SectorManagement`` rules address original farm turbine indices or names,
regardless of the changing downwind order. Rules modify only selected
state-turbine entries whose values satisfy inclusive lower and exclusive upper
range bounds; periodic direction ranges may cross north.

``PowerMask`` boosts ramp smoothly across ``P_lim`` power units below rated
power, reaching ``FV.MAX_P`` at rated power. ``P_lim`` must be finite and
nonnegative; zero requests an abrupt boost above rated power. NaN masks leave
power and thrust unchanged. Thrust saturates at the low-induction Betz optimum
(``a = 1/3``, ``CT = 8/9``) when a boost exceeds the calibrated theoretical
power maximum. Requested boosted power is retained, so this is an engineering
thrust approximation rather than a prediction of attainable turbine power.

.. _farm-controllers:

Farm controllers
----------------
Depending on the setup of the wind farm, each turbine model applies to a subset of the turbines in the farm.
Farm controllers are responsible for running the turbine models accordingly, and optionally depending on
conditions.

The list of available farm controller classes can be found
:doc:`here in the API <_autoapi/foxes/models/farm_controllers/index>`. For example:

* :class:`BasicFarmController<foxes.models.farm_controllers.basic.BasicFarmController>`: Calls turbine models based on their order of appearance, moving those with `pre_rotor` flag to the front.
* :class:`OpFlagController<foxes.models.farm_controllers.op_flag.OpFlagController>`: Requires an operation flag farm variable for each state and turbine, and only runs turbine models for switched-on turbines.

The controller is selected when constructing the algorithm object, via the parameter
`farm_controller`. If ommited, this defaults to `basic_ctrl` which corresponds to
the `BasicFarmController` in the model book.

.. _ground-models:

Ground models
-------------
Add ground effects to the wake calculation, for example the reflection from horizontal planes.

The list of available ground model classes can be found
:doc:`here in the API <_autoapi/foxes/models/ground_models/index>`. The following models are
accessible from the default model book:

* `no_ground`: Does not add any ground effects.
* `ground_mirror`: Adds wake reflection from a horizontal plane at zero height.
* `blh_mirror_h<height>`: Adds wake reflections from two horizontal planes, one at the ground and one at the specified height.

Ground models can be selected globally for all wake models, by passing the model
name to the `ground_models` argument of the algorithm constructor. Alternatively, a
dictionary mapping of wake model names to ground model names can be used, cf. the rules
for partial wakes model selections.

.. _point-models:

Point models
------------
Calculate point-based data during the evaluation of `algo.calc_points()`,
or as a modification of ambient states.

Point models can be added to ambient states objects, simply by the `+` operation.

The list of available point models can be found in the :doc:`API <_autoapi/foxes/models/point_models/index>`.
For example:

* :class:`WakeDeltas<foxes.models.point_models.wake_deltas.WakeDeltas>`: Subtract backgrounds from waked results.
* :class:`TKE2TI<foxes.models.point_models.tke2ti.TKE2TI>`: Compute `TI` from turbulent kinetic energy data, as for example provided by mesoscale simulations.

.. _vertical-profiles:

Vertical profiles
-----------------
Analytical vertical profiles transform uniform ambient states into height dependent inflow.

The list of available vertical profiles can be found in the :doc:`API <_autoapi/foxes/models/vertical_profiles/index>`.
they can be added to uniform ambient states as in the following example, here for
a Monin-Obukhof dependent log-profile:

    .. code-block:: python

        states = foxes.input.states.StatesTable(
            data_source="abl_states_6000.csv.gz",
            output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO, FV.MOL],
            var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti", FV.MOL: "mol"},
            fixed_vars={FV.RHO: 1.225, FV.Z0: 0.05, FV.H: 100.0},
            profiles={FV.WS: "ABLLogWsProfile"},
        )

Notice the required variable `FV.H`, denoting the reference height of the
provided wind data, as well as roughness length `FV.Z0` and Monin-Obukhof length `FV.MOL`.
