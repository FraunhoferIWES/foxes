
.. image:: ../../Logo_FOXES.svg
    :align: center

.. versionadded:: 1.6
    Parallel results writing directly by the `calc_farm` and `calc_points` functions of the algorithm,
    optionally without ever constructing the complete dataset in memory. If the input consists of multiple files,
    the output can be written into consistent multiple files as well.
    
.. versionadded:: 1.6
    Direct support for WRF data fields in `NEWA <https://map.neweuropeanwindatlas.eu/>`_ format, via
    the new ambient states class :ref:`NEWAStates<foxes.input.states.NEWAStates>`.

.. versionadded:: 1.6
    Support for wind farm layouts in (longitude, latitude) coordinates, will
    be automatically converted into the globally set UTM zone. The new layout input functions 
    :ref:`add_from_wrf<foxes.input.farm_layout.add_from_wrf>` for WRF wind farm input folders and 
    :ref:`add_from_eww<foxes.input.farm_layout.add_from_eww>` for farm input in EuroWindWakes format
    are always based on (lon, lat) coordinates.

.. versionchanged:: 1.6
    The :ref:`Iterative <foxes.algorithms.Iterative>` algorithm now reduces the
    target states to the subset of non-converged states in each iteration, which
    can significantly reduce computation time.

.. versionadded:: 1.6
    Support for Python 3.14.

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
optimization capabilities invoke the `foxes-opt <https://github.com/FraunhoferIWES/foxes-opt>`_
package which as well supports vectorization and parallelization.

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

        parameter_files

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
