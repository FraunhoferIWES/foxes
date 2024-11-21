foxes.input
===========
Classes and functions that describe user input data.

    .. table:: 
        :widths: auto

        =======================================  =============================================================
        Package                                  Description
        =======================================  =============================================================
        :ref:`foxes.input.farm_layout`           Functions for adding turbines to the wind farm.
        :ref:`foxes.input.states`                Atmospheric input states.
        :ref:`foxes.input.states.create`         Functions for the creation of ambient states from parameters.
        :ref:`foxes.input.yaml`                  Runs *foxes* via parameter input yaml files, for *foxes_yaml*
        :ref:`foxes.input.yaml.windio`           An interface to WindIO yaml files, via *foxes_windio*
        =======================================  =============================================================

foxes.input.farm_layout
-----------------------
This package contains functions that can be used to add
wind turbines to the wind farm.

    .. python-apigen-group:: input.farm_layout

foxes.input.states
------------------
All ambient user input states classes can be found here.

    .. python-apigen-group:: input.states

foxes.input.states.create
-------------------------
Functions for the creation of ambient states from parameters.

    .. python-apigen-group:: input.states.create

foxes.input.yaml
------------------
Runs *foxes* via parameter input yaml files, used by the command line application
*foxes_yaml*.

    .. python-apigen-group:: input.yaml
        

foxes.input.yaml.windio
-----------------------
Interface to WindIO yaml input files, used by the command line application
*foxes_windio*.

    .. python-apigen-group:: input.yaml.windio