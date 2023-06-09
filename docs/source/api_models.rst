foxes.models
============
This package contains all concrete model classes.

foxes.models.farm_controllers
-----------------------------
The wind farm controller models. They are responsible
for calling turbine models in the correct order and for
the relevant subset of wind turbines.

    .. python-apigen-group:: models.farm_controllers

foxes.models.farm_models
------------------------
The wind farm models, ie., turbine models that
are applied to all turbines of the wind farm.

    .. python-apigen-group:: models.farm_models

foxes.models.partial_wakes
--------------------------
The partial wake models. They are responsible for 
computing rotor effective wake deltas.

    .. python-apigen-group:: models.partial_wakes

foxes.models.point_models
-------------------------
The point models. They compute state-point data for 
given points of interest.

    .. python-apigen-group:: models.point_models

foxes.models.rotor_models
-------------------------
The rotor models. They compute rotor effective ambient data
from the ambient input states.

    .. python-apigen-group:: models.rotor_models

foxes.models.turbine_models
---------------------------
The turbine models. They compute state-turbine data based on 
the currently available and model provided data.

    .. python-apigen-group:: models.turbine_models

foxes.models.turbine_types
--------------------------
The turbine type models. These are turbine models that represent
the wind turbine machine, i.e, they specify rotor diameter, hub 
height and compute power and thrust.

    .. python-apigen-group:: models.turbine_types

foxes.models.vertical_profiles
------------------------------
The vertical profile models. They compute height dependent data
in one dimension, e.g., wind speed profiles.

    .. python-apigen-group:: models.vertical_profiles

foxes.models.wake_frames
------------------------
The wake frame models. They define the curves along which the wakes
propagate.

    .. python-apigen-group:: models.wake_frames

foxes.models.wake_models
------------------------
The wake models. They compute wake deltas from source turbines at 
given evaluation points.

    .. toctree::
        :maxdepth: 2

        api_wake_models

foxes.models.wake_superpositions
--------------------------------
The wake superposition models. These models compute net wake effects 
from individual wake delta results. Note that wake models can but do not 
neccessarily have to make use of wake superposition models.

    .. python-apigen-group:: models.wake_superpositions
