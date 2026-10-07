
.. image:: ../../Logo_FOXES.svg
    :align: center

.. versionadded:: 1.9.7
    Support for Python 3.14.

.. versionadded:: 1.9.7
    ``FarmLayoutOutput`` can render physical rotor radii and override farm
    boundaries for figures without changing numerical farm geometry.

Welcome to FOXES
================

*Farm Optimization and eXtended yield Evaluation Software*

*FOXES* is a modular Python package for wind-farm and wake modelling by
Fraunhofer IWES. Typical applications include:

* Wind farm optimization, e.g. layout optimization or wake steering,
* Wind farm post-construction analysis,
* Wake model studies, comparison and validation,
* Wind farm simulations invoking complex model chains.

FOXES uses vectorized calculations and supports local or distributed execution.
Wind-farm optimization is provided by the separate
`foxes-opt <https://github.com/FraunhoferIWES/foxes-opt>`_ package.

Source code repository (and issue tracker):
    https://github.com/FraunhoferIWES/foxes

Please report issues in the repository's issue tracker.

License
-------
    MIT_

.. _MIT: https://github.com/FraunhoferIWES/foxes/blob/main/LICENSE

Contents
--------

.. toctree::
    :maxdepth: 2

    citation
    installation
    overview
    inputs
    models
    notebooks/parallelization
    parameter_files
    examples
    optimization
    api
    notebooks/data
    testing
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

Development of *FOXES* and its internal predecessors was supported by:

* BMWK projects *Smart Wind Farms* (0325851B), *GW-Wakes* (0325397B), and
    *X-Wakes* (03EE3008A)
* BMBF project *H2Digital* (03SF0635)
* Horizon Europe projects *FLOW* (101084205) and *AIRE* (101083716)
