
.. image:: ../../Logo_FOXES.svg
    :align: center

Welcome to FOXES
================

*Farm Optimization and eXtended yield Evaluation Software*

*FOXES* is a modular wind farm and wake modelling code written in Python 
by Fraunhofer IWES. It has many applications, for example

* Wind farm optimization, e.g. layout optimization or wake steering,
* Wind farm post-construction analysis,
* Wake model studies, comparison and validation,
* Wind farm simulations invoking complex model chains.

The calculation is fully vectorized and its fast performance is owed to `dask <https://www.dask.org/>`_.
Also the parallelization on local or remote clusters is enabled via `dask`. The wind farm
optimization capabilities invoke the `iwopy <https://github.com/FraunhoferIWES/iwopy>`_
package which as well supports vectorization.

**Quick Start**::

    pip install foxes

Source code repository (and issue tracker):
    https://github.com/FraunhoferIWES/foxes

Contact (please report code issues under the github link above):
    :email:`Jonas Schmidt <jonas.schmidt@iwes.fraunhofer.de>`
    
License:
    MIT_

.. _MIT: https://github.com/FraunhoferIWES/foxes/blob/main/LICENSE

Contents:
    .. toctree::
        :maxdepth: 2
    
        installation

    .. toctree::
        :maxdepth: 2

        examples
        
    .. toctree::
        :maxdepth: 1

        api

    .. toctree::
        :maxdepth: 2

        history

Contributing
============

#. Fork *foxes* on *github*.
#. Create a branch (`git checkout -b new_branch`)
#. Commit your changes (`git commit -am "your awesome message"`)
#. Push to the branch (`git push origin new_branch`)
#. Create a pull request

Acknowledgements
================

The development of *foxes* and its predecessors *flapFOAM* and *flappy* (internal - non public) 
has been supported through multiple publicly funded research projects. We acknowledge in particular 
the funding by the Federal Ministry of Economic Affairs and Climate Action (BMWK) through the p
rojects *Smart Wind Farms* (grant no. 0325851B), *GW-Wakes* (0325397B) and *X-Wakes* (03EE3008A) 
as well as the funding by the Federal Ministry of Education and Research (BMBF) in the framework 
of the project *H2Digital* (03SF0635).
