foxes.models
============
This package contains all concrete model classes.

    .. table:: 
        :widths: auto

        =======================================  ============================================================
        Package                                  Description
        =======================================  ============================================================
        :ref:`foxes.models.farm_controllers`     Wind farm controller models. They execute turbine models.
        :ref:`foxes.models.farm_models`          Wind farm models, i.e., turbine models for all turbines.
        :ref:`foxes.models.partial_wakes`        Partial wake models, computing rotor effective wake deltas.
        :ref:`foxes.models.point_models`         Point models, calculating results at points of interest.
        :ref:`foxes.models.rotor_models`         Rotor models, computing rotor effective ambient results.
        :ref:`foxes.models.turbine_models`       Turbine models, calculating data at turbines.
        :ref:`foxes.models.turbine_types`        Turbine types, providing power and thrust.
        :ref:`foxes.models.vertical_profiles`    Vertical profiles, for atmospheric input.
        :ref:`foxes.models.axial_induction`      Axial induction models, computing axial induction factors.
        :ref:`foxes.models.wake_frames`          Wake frames, defining the wake propagation.
        :ref:`foxes.models.wake_models`          Wake models, computing variable changes due to wakes.
        :ref:`foxes.models.wake_superpositions`  Wake superposition models, evaluating multiple wakes deltas.
        =======================================  ============================================================

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

foxes.models.axial_induction
----------------------------
The axial induction models, basically providing a function `ct2a`.

    .. python-apigen-group:: models.axial_induction

foxes.models.wake_frames
------------------------
The wake frame models. They define the curves along which the wakes
propagate.

    .. python-apigen-group:: models.wake_frames

foxes.models.wake_models
------------------------
The wake models. They compute wake deltas due to source turbines at 
any set of evaluation points.

    .. toctree::
        :maxdepth: 2

        api_wake_models

foxes.models.wake_superpositions
--------------------------------
The wake superposition models. These models compute net wake effects 
from individual wake delta results. Note that wake models can but do not 
neccessarily have to make use of wake superposition models.

    .. python-apigen-group:: models.wake_superpositions
