foxes.algorithms
================
Algorithms manage the calculation of farm and point results.

    .. table:: 
        :widths: auto

        ========================== ====================================================
        Algorithm                  Description
        ========================== ====================================================
        :ref:`Downwind algorithm`  Orders turbines in downwind direction, single sweep.
        :ref:`Iterative algorithm` Iterates until convergence has been reached.
        ========================== ====================================================

Downwind algorithm
------------------
Orders turbines in downwind direction, then applies a single sweep.

    .. python-apigen-group:: algorithms.downwind

    .. python-apigen-group:: algorithms.downwind.models

Iterative algorithm
-------------------
Iterates wake and turbine model evaluations until convergence has been reached.

    .. python-apigen-group:: algorithms.iterative

    .. python-apigen-group:: algorithms.iterative.models
