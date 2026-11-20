foxes.core
==========
Contains core functionality and abstract base classes.

``Output`` declares chunk input-variable requirements and the
``calculate``, ``update_results`` and ``finalize_results`` lifecycle.
``OutputList`` composes outputs and merges their requirements. Concrete adopted
outputs live in ``foxes.models.outputs``; see :doc:`api_models`.

.. toctree::
    :maxdepth: 2

    _autoapi/foxes/core/index
