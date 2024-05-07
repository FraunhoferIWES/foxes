Models
======

Any *foxes* run relies on a number of model choices by the user:

* :ref:`Rotor models`: Evaluate the flow field at the rotor and compute ambient rotor equivalent quantities, for example the rotor averaged background wind speed.
* :ref:`Turbine types`: Define rotor diameter and hub height, and provide thrust coefficient and power yield depending on rotor equivalent quantities. 
* :ref:`Wake frames`: Determine the path of wake propagation and the local coordinate system that follows the centreline of the wake.
* :ref:`Wake models`: Compute wake deltas for flow quantities in the wake. 
* *partial wakes model*: Evaluates rotor disc averages of wake effects, i.e., the partial wakes model computes the rotor effective wake deltas. 
* *turbine models*: Each wind turbine within the wind farm can have individual turbine model choices. For each state and turbine, those compute data from currently existing data. For example, depending on the rotor effective wind speed and wind direction, a turbine model might correct the power and thrust coefficients that were provided by the turbine type model (wind sector management).
* *point models* (optional): They calculate point-based data, like those from the ambient input states. For example, if WRF based input data provides a time series of turbulent kinetic energy (TKE) this can be translated into turbulence intensity (TI) by a point model, as required by other models in `foxes`.

All concrete models are stored in the so-called :code:`ModelBook` object under 
a name string, see :ref:`this example<The model book>`.

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

Wake frames
-----------
Wake frames determine the path of wake propagation, for example parallel to the 
wind direction at the rotor, or along a streamline, and the local coordinate system 
that follows the centreline of the wake.

The available wake frame classes are listed :ref:`here<foxes.models.wake_frames>`
in the API. The :ref:`default model book<The model book>` contains many pre-defined wake
frames, for example:

* `rotor_wd`: Straight wakes, following the wind direction measured at the centre of the wake causing rotor.
* `yawed`, `yawed_k002`, `yawed_k004`: Wake bending due to yaw misalignment of the rotor, as represented by the `YAWM` variable. See :ref:`this example<Yawed rotor wakes>`.  
* `streamlines_X`: Streamline (or streaklines) following steady-state wakes, for a virtual time step of `X = 1, 5, 10, 50, 100, 500` seconds. See :ref:`this example<Heterogeneous flow>`.
* `timelines`, `timelines_X`: Dynamic flow following wakes for spatially homogeneous wind data, optionally with time step of `X = 1s, 10s, 30s, 1min, 10min, 30min`. See :ref:`this example<Dynamic wakes 1>`.
* `seq_dyn_wakes`, `seq_dyn_wakes_X`: Sequential state evaluation (caution: slow, no state chunking), optionally with time step of `X = 1s, 10s, 30s, 1min, 10min, 30min`. See :ref:`this example<Dynamic wakes 2>`.

Wake models
-----------
Wake models compute wake deltas for flow quantities in the wake. Wind speed deficits and turbulence 
intensity deltas are often computed by two separate wake models, but could also stem from a single model. 
Wake superposition is part of the responsibility of the wake model.

The wake model classes can be found :ref:`here in the API<foxes.models.wake_models>`.
They are structured according to their mathematical nature:
