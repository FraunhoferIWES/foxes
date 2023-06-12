foxes.opt
==========
Wind farm optimization within `foxes` is run
via the `foxes.opt` sub-package. This internally
makes use of the external `iwopy` package, whose
documentation can be found
`here <https://fraunhoferiwes.github.io/iwopy.docs/index.html>`_.

    .. table:: 
        :widths: auto

        =======================================  ============================================================
        Package                                  Description
        =======================================  ============================================================
        :ref:`foxes.opt.core`                    Abstract base classes and core functionality.
        :ref:`foxes.opt.problems`                Wind farm optimization problems.
        :ref:`foxes.opt.objectives`              Objectives for wind farm optimization problems.
        :ref:`foxes.opt.constraints`             Constraints for wind farm optimization problems.
        =======================================  ============================================================

foxes.opt.core
--------------
Contains core functionality and abstract base classes.

    .. python-apigen-group:: opt.core

foxes.opt.problems
------------------
Wind farm optimization problems.

    .. toctree::
        api_opt_problems

foxes.opt.objectives
--------------------
Objectives for wind farm optimization problems.

    .. python-apigen-group:: opt.objectives

foxes.opt.constraints
---------------------
Constraints for wind farm optimization problems.

    .. python-apigen-group:: opt.constraints
