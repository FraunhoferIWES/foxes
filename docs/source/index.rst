
.. image:: ../../Logo_FOXES.svg
    :align: center

.. versionchanged:: 1.0
    User-selectable :ref:`Parallelization` via the new `Engines`, replacing `Runners`. 
    The default is now based on `concurrent.futures <https://docs.python.org/3/library/concurrent.futures.html>`_ and comes with a speedup. 
    Also `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ is now supported, for simplified
    multi-node computations.

.. versionadded:: 1.0
    New wake frame :ref:`DynamicWakes<Dynamic Wakes 1>`: Chunk-based vectorized dynamic
    wakes for any kind of inflow 

.. versionadded:: 1.0
    New inflow class :ref:`OnePointFlowStates<Dynamic Wakes 1>`: 
    Heterogeneous inflow fields extrapolated from data at a single point

.. versionadded:: 1.0
    New turbine type :ref:`FromLookupTable<foxes.models.turbine_types.FromLookupTable>`: 
    Interpolates power and thrust coefficient from lookup table input data

Welcome to FOXES
================

*Farm Optimization and eXtended yield Evaluation Software*

*FOXES* is a modular wind farm and wake modelling code written in Python 
by Fraunhofer IWES. It has many applications, for example

* Wind farm optimization, e.g. layout optimization or wake steering,
* Wind farm post-construction analysis,
* Wake model studies, comparison and validation,
* Wind farm simulations invoking complex model chains.

The fast performance of *foxes* is owed to vectorization and parallelization,
and it is intended to be used for large wind farms and large timeseries inflow data.
The parallelization on local or remote clusters is supported, based on 
`mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ or
`dask.distributed <https://distributed.dask.org/en/stable/>`_.
The wind farm
optimization capabilities invoke the `iwopy <https://github.com/FraunhoferIWES/iwopy>`_
package which as well supports vectorization.

Source code repository (and issue tracker):
    https://github.com/FraunhoferIWES/foxes

Please report code issues under the github link above.
    
License
-------
    MIT_

.. _MIT: https://github.com/FraunhoferIWES/foxes/blob/main/LICENSE

Contents
--------
    .. toctree::
        :maxdepth: 2
    
        citation

    .. toctree::
        :maxdepth: 2
    
        installation

    .. toctree::
        :maxdepth: 1

        overview

    .. toctree::
        :maxdepth: 2

        inputs

    .. toctree::
        :maxdepth: 2

        models

    .. toctree::
        :maxdepth: 2

        notebooks/parallelization

    .. toctree::
        :maxdepth: 2

        examples

    .. toctree::
        :maxdepth: 2

        optimization

    .. toctree::
        :maxdepth: 1

        api

    .. toctree::
        :maxdepth: 2

        notebooks/data

    .. toctree::
        :maxdepth: 1

        testing

    .. toctree::
        :maxdepth: 1

        CHANGELOG

Contributing
------------

#. Fork *foxes* on *github*.
#. Create a branch (`git checkout -b new_branch`)
#. Commit your changes (`git commit -am "your awesome message"`)
#. Push to the branch (`git push origin new_branch`)
#. Create a pull request `here <https://github.com/FraunhoferIWES/foxes/pulls>`_

Acknowledgements
----------------

The development of *foxes* and its predecessors *flapFOAM* and *flappy* (internal - non public) 
has been supported through multiple publicly funded research projects. We acknowledge in particular 
the funding by the Federal Ministry of Economic Affairs and Climate Action (BMWK) through the p
rojects *Smart Wind Farms* (grant no. 0325851B), *GW-Wakes* (0325397B) and *X-Wakes* (03EE3008A) 
as well as the funding by the Federal Ministry of Education and Research (BMBF) in the framework 
of the project *H2Digital* (03SF0635). We furthermore acknowledge funding by the Horizon Europe 
project FLOW (Atmospheric Flow, Loads and pOwer for Wind energy - grant id 101084205).
