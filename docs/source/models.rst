Models
======

Model types
-----------

The results of *foxes* runs depend on a number of model choices by the user:

* :ref:`Rotor models`: Evaluate the flow field at the rotor and compute ambient rotor equivalent quantities.
* :ref:`Turbine types`: Define rotor diameter and hub height, and provide thrust coefficient and power yield depending on rotor equivalent quantities. 
* :ref:`Wake models`: Compute wake deltas for flow quantities in the wake of a turbine.
* :ref:`Wake frames`: Determine the path of wake propagation and the local coordinate system that follows the centreline of the wake.
* :ref:`Partial wakes`: Compute rotor disc averages of wake effects, i.e., the partial wakes models calculate the rotor effective wake deltas. 
* :ref:`Turbine models`: Each wind turbine within the wind farm can have individual turbine model choices. For each state and turbine, those compute data from currently existing data. 
* :ref:`Ground models`: Add ground effects to the wake calculation, for example the reflection from horizontal planes.
* :ref:`Point models`: Calculate point-based data during the evaluation of `algo.calc_points()`, or as a modification of ambient states., like those from the ambient input states. 
* :ref:`Vertical profiles`: Analytical vertical profiles transform uniform ambient states into height dependent inflow.

All concrete models are stored in the so-called :code:`ModelBook` object under 
a name string (or a name string template that is provided by a so-called model factory), 
see :ref:`this example<The model book>`.

Rotor models
------------
Rotor models evaluate the flow field at the rotor and compute ambient rotor equivalent quantities, for example the rotor averaged 
background wind speed. The list of available models and also rotor point visualizations can be found :ref:`in this overview<Rotor model visualizations>`.

Turbine types
-------------
Turbine type models define rotor diameter and hub height, and provide thrust coefficient and power yield depending on rotor equivalent quantities. 
They are either chosen from the provided :ref:`static data<Power and thrust curves>` by their model name (e.g. *DTU10MW*), 
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
can be found :ref:`here<foxes.models.turbine_types>` in the API. 

Wake models
-----------
Wake models compute wake deltas for flow quantities in the wake. Wind speed deficits and turbulence 
intensity deltas are often computed by two separate wake models, but could also stem from a single model. 

The wake model classes can be found :ref:`here in the API<foxes.models.wake_models>`.
They are organized into three sub-packages, according to their purpose and target variables: 

* :ref:`wind<foxes.models.wake_models.wind>`: Wind deficit models, computing negative deltas for the wind speed variable `WS`,
* :ref:`ti<foxes.models.wake_models.ti>`: Positive wake deltas acting on the variable `TI`, modelling the turbulence increase within the wake region,
* :ref:`induction<foxes.models.wake_models.induction>`: Individual turbine induction models acting as wind speed deltas, which, in combination, model wind farm blockage effects.

Note that `wind` and `ti` wake models affect downstream turbines, while `induction` models 
mainly affect upstream and stream-orthogonal turbines. During calulations, a list of
wake models is expected, so in principle, a wind deficit model, a TI wake model and a turbine
induction model can be combined. If an induction model is included in the
list of model selections, the :ref:`Iterative algorithm` has to be applied.

All wake model classes are implemented according to their mathematical nature, i.e.,
if applicable, they are derived from one of the following types:

* :ref:`DistSlicedWakeModel<foxes.models.wake_models.DistSlicedWakeModel>`: The wake delta depends on the wake frame coordinate `x` differently than on `(y, z)`, e.g., the `x` dependency can be factorized.
* :ref:`AxisymmetricWakeModel<foxes.models.wake_models.AxisymmetricWakeModel>`: Dist-sliced wake with axial symmetry, i.e., the wake can be described by `x` and a radial wake frame coordinate `r`.
* :ref:`GaussianWakeModel<foxes.models.wake_models.GaussianWakeModel>`: Axisymmetric wake that follows a Gaussian function, where the standard deviation `sigma(x)` depends on `x` only.
* :ref:`TopHatWakeModel<foxes.models.wake_models.TopHatWakeModel>`: Axisymmetric wake that is independent of `r` within the top-hat shape, and zero outside.

The reasoning behind this is that the partial wakes models can then
build upon the underlying shape of the wake.

Wake superposition is part of the responsibility of the wake model. Most models expect
a choice of the underlying :ref:`wake superposition model<foxes.models.wake_superpositions>`
in their constructor, in terms of their respective name in the :ref:`model book<The model book>`.
Examples are `ws_linear` for linear wind deficit superposition, or `ti_quadratic`
for quadratic TI wake increase superposition.

The list of wake model name templates in the :ref:`default model book<The model book>` is long,
but that is mainly due to variations of various constructor argument choices. Typical examples are

* `Jensen_<superposition>_[wake_k]`: The classic Jensen wind deficit model,
* `Bastankhah2014_<superposition>_[wake_k]`: The Gaussian wind deficit model by `Bastankhah and Porté-Agel from 2014 <https://doi.org/10.1016/j.renene.2014.01.002>`_,
* `Bastankhah2016_<superposition>_[wake_k]`: The wind deficit model by `Bastankhah and Porté-Agel from 2016 <https://doi.org/10.1017/jfm.2016.595>`_,
* `TurbOPark_<superposition>_[wake_k]`: The Gaussian wind deficit model by `Pedersen et al. from 2022 <https://iopscience.iop.org/article/10.1088/1742-6596/2265/2/022063/pdf>`_,
* `CrespoHernandez_<superposition>_[wake_k]`: The top-hat TI addition wake model by `Crespo and Hernandez from 1996 <https://doi.org/10.1016/0167-6105(95)00033-X>`_,
* `IECTI2019_<superposition>`: The top-hat TI addition wake model by `Frandsen from 2019 <http://orbit.dtu.dk/files/3750291/2009_31.pdf>`_.

Note that in all above cases, the `superposition` parameter is 
any of the available :ref:`wake superposition models<foxes.models.wake_superpositions>` for wind speed or TI, depending on the model.
Here the convention is that you write `linear` for the choice `ws_linear` or `ti_linear`, etc., depending if the wake model targets wind speed or TI
(cf. the :ref:`model book<The model book>` example). 

The `[wake_k]` part of the model name can be replaced by one of the following patterns:

* `k<k>`, where `<k>` is to be replaced by the value for the wake growth factor `k`, with dot-skipping convention (e.g. `004` for the value `0.04`, etc.) 
* `ka<ka>`, where `<ka>` is to be replaced by the value for `ka` in `k = ka * TI`, with dot-skipping convention (e.g. `004` for the value `0.04`, etc.) 
* `ambka<ka>`, where `<ka>` is to be replaced by the value for `ka` in `k = ka * AMB_TI`, with dot-skipping convention (e.g. `004` for the value `0.04`, etc.) 
* `ka<ka>_kb<kb>`, where `<ka>` and `<kb>` are to be replaced by the values for `ka` and `kb` in `k = ka * TI + kb`, both with dot-skipping convention (e.g. `004` for the value `0.04`, etc.) 
* `ambka<ka>_kb<kb>`, where `<ka>` and `<kb>` are to be replaced by the values for `ka` and `kb` in `k = ka * AMB_TI + kb`, both with dot-skipping convention (e.g. `004` for the value `0.04`, etc.) 
* nothing, e.g. `Bastankhah2014_linear`, which searches the value for `k` in the list of available farm data. This is intended to be used whenever a turbine model computes the `k` values, typically the the :ref:`kTI<foxes.models.turbine_models.kTI>` turbine model, or an optimizer.

Examples for valid wake model choices are:

* `Jensen_quadratic_k0075`
* `Bastankhah2014_linear_ka02_kb0012`
* `Bastankhah2016_linear_lim_ambka04`
* `TurbOPark_quadratic_loc_k004`
* `CrespoHernandez_max_ka0213_kb003`
* `Bastankhah2014_linear`

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
:ref:`here in the API<foxes.models.wake_frames>`. The :ref:`default model book<The model book>` 
contains many pre-defined wake frames, for example:

* `rotor_wd`: Straight wakes, following the wind direction measured at the centre of the wake causing rotor.
* `yawed_[wake_k]`: Wake bending due to yaw misalignment of the rotor, as represented by the `YAWM` variable. See :ref:`this example<Yawed rotor wakes>`.  
* `streamlines_<step>`: Streamline (or streaklines) following steady-state wakes, for a virtual time step of `step` seconds. See :ref:`this example<Heterogeneous flow>`.
* `timelines`, `timelines_<dt>`: Dynamic flow following wakes for spatially homogeneous wind data, optionally with time step `dt`, e.g. `dt=10s` or `dt=1min`, or other values with one of those two units. See :ref:`this example<Dynamic wakes 1>`.
* `seq_dyn_wakes`, `seq_dyn_wakes_<dt>`: Sequential state evaluation (caution: slow, no state chunking), optionally with time step `dt`, e.g. `dt=10s` or `dt=1min`, or other values with one of those two units. See :ref:`this example<Dynamic wakes 2>`.

The `yawed` wake frame is based on the wind deficit model by `Bastankhah and Porté-Agel from 2016 <https://doi.org/10.1017/jfm.2016.595>`_,
and when it is combined with the corresponding wake model `Bastankhah2016_<superposition>_[wake_k]` it picks up all
model parameters from that instance.  For other wake models, the parameters can either be specified by adding a new instance
of the :ref:`YawedWakes<foxes.models.wake_frames.YawedWakes>` class with the desired constructor call, or the defaults will 
be taken. The `k` wake growth parameter is either specified by the rules described above in the :ref:`Wake models` section, or it will be
picked automatically from the first wake model in the wake model list given to the algorithm.

Partial wakes
-------------
Partial wakes models compute rotor disc averages of wake effects, i.e., 
the partial wakes models calculate the rotor effective wake deltas. 

Some of the partial wakes models make use of the mathematical structure of 
the associated wake model:

* :ref:`PartialCentre<foxes.models.partial_wakes.PartialCentre>`: Only evaluate wakes at rotor centres. This is fast, but not accurate.
* :ref:`RotorPoints<foxes.models.partial_wakes.RotorPoints>`: Evaluate the wake model at exactly the rotor points, then take the average of the combined result. For large number of rotor points this is accurate, but potentially slow.
* :ref:`PartialTopHat<foxes.models.partial_wakes.PartialTopHat>`: Compute the overlap of the wake circle with the rotor disc. This is mathematically exact and fast, but limited to wakes with top-hat shapes.
* :ref:`PartialAxiwake<foxes.models.partial_wakes.PartialAxiwake>`: Compute the numerical integral of axi-symmetric wakes with the rotor disc. This needs less evaluation points than grid-type wake averaging.
* :ref:`PartialSegregated<foxes.models.partial_wakes.PartialSegregated>`: Abstract base class for segregated wake averaging, which means adding the averaged wake to the averaged background result (in contrast to `RotorPoints`).
* :ref:`PartialGrid<foxes.models.partial_wakes.PartialGrid>`: Segregated partial wakes evaluated at points of a :ref:`grid-type rotor<GridRotor>` (which is usually not equal to the selected rotor model).

In the :ref:`default model book<The model book>`, concrete instances of the above partial wakes models
can be found under the names

* `centre`: The centre point model,
* `rotor_points`: The rotor points model,
* `top_hat`: The top-hat model,
* `axiwake<n>`: The axiwake model, with `n` representing the number of steps for the discretization of the integral over each downstream rotor,
* `grid<n2>`: The grid model with `n2` representing the number of points in a regular square grid.

Partial wakes are now chosen when costructing the algorithm object.
There are several ways of specifying partial wakes model choices for 
the selected wake models:

* by a dictionary, which maps wake model names to model choices (or default choices, if not found),
* or by a list, where the mapping to the wake models is in order of appearance, 
* or by a string, in which case all models are either mapped to the given model, or, if that fails with `TypeError`, to their defaults,
* or by `None`, which means all models are mapped to the default choice.

A verification of the different partial wakes models 
is carried out in this example: :ref:`Partial wakes verification`
All types approach the correct rotor average for high point
counts, but with different efficiency.

Turbine models
--------------
Each wind turbine within the wind farm can have individual turbine model choices. 
For each state and turbine, those compute data from currently existing data. 

The list of available turbine model classes can be found 
:ref:`here in the API<foxes.models.turbine_models>`. For example:

* :ref:`kTI<foxes.models.turbine_models.kTI>`: Computes the wake expansion coefficient `k` as a linear function of `TI`: `k = kb + kTI * TI`. All models that do not specify `k` explicitly (i.e, `k=None` in the constructor), will then use this result when computing wake deltas.
* :ref:`SetFarmVars<foxes.models.turbine_models.SetFarmVars>`: Set any farm variable to any state-turbine data array, or sub-array (nan values are ignored), either initially (`pre_rotor=True`) or after the wake calculation.
* :ref:`PowerMask<foxes.models.turbine_models.PowerMask>`: Curtail or boost the turbine by re-setting the maximal power of the turbine, see :ref:`this example<Power mask>`.
* :ref:`SectorManagement<foxes.models.turbine_models.SectorManagement>`: Modify farm variables if wind speed and/or wind direction values are within certain ranges, see :ref:`this example<Wind sector management>`.
* :ref:`YAW2YAWM<foxes.models.turbine_models.YAW2YAWM>` and :ref:`YAWM2YAW<foxes.models.turbine_models.YAWM2YAW>`: Compute absolute yaw angles from yaw misalignment, and vice-versa.
* :ref:`Calculator<foxes.models.turbine_models.Calculator>`: Apply any user-written function that calculates values of farm variables.
* :ref:`LookupTable<foxes.models.turbine_models.LookupTable>`: Use a lookup-table for the computation of farm variables.


Ground models
-------------
Add ground effects to the wake calculation, for example the reflection from horizontal planes.

The list of available ground model classes can be found 
:ref:`here in the API<foxes.models.ground_models>`. The following models are 
accessible from the :ref:`default model book<The model book>`:

* `no_ground`: Does not add any ground effects.
* `ground_mirror`: Adds wake reflection from a horizontal plane at zero height.
* `blh_mirror_h<height>`: Adds wake reflections from two horizontal planes, one at the ground and one at the specified height.

Ground models can be selected globally for all wake models, by passing the model
name to the `ground_models` argument of the algorithm constructor. Alternatively, a 
dictionary mapping of wake model names to ground model names can be used, cf. the rules
for :ref:`partial wakes model selections<Partial wakes>`.

Point models
------------
Calculate point-based data during the evaluation of `algo.calc_points()`, 
or as a modification of ambient states.

Point models can be added to ambient states objects, simply by the `+` operation.

The list of available point models can be found :ref:`here in the API<foxes.models.point_models>`.
For example:

* :ref:`WakeDeltas<foxes.models.point_models.WakeDeltas>`: Subtract backgrounds from waked results.
* :ref:`TKE2TI<foxes.models.point_models.TKE2TI>`: Compute `TI` from turbulent kinetic energy data, as for example provided by mesoscale simulations.

Vertical profiles
-----------------
Analytical vertical profiles transform uniform ambient states into height dependent inflow.

The list of available vertical profiles can be found :ref:`here in the API<foxes.models.vertical_profiles>`.
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