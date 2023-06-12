API
===

foxes
-----
The top-level package provides direct access to some 
of the main classes and functions.

All objects listed here can be reached directly
via the *foxes* package, e.g. *foxes.WindFarm*.

    .. python-apigen-group:: foxes

foxes.core
----------
Contains core functionality and abstract base classes.

    .. python-apigen-group:: core

foxes.input
-----------
Contains classes and functions that describe user input data.

    .. toctree::
        :maxdepth: 2

        api_input

foxes.models
------------
This package contains all concrete model classes.

    .. toctree::
        :maxdepth: 2

        api_models

foxes.utils
------------
Utilities and helper functions that are not *foxes* specific.

    .. toctree::
        :maxdepth: 2

        api_utils

foxes.opt
---------
Wind farm optimization within `foxes` is run
via the `foxes.opt` sub-package. This internally
makes use of the external `iwopy` package, whose
documentation can be found
`here <https://fraunhoferiwes.github.io/iwopy.docs/index.html>`_.

    .. toctree::
        :maxdepth: 2

        api_opt