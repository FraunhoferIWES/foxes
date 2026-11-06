Examples
========

Python scripts
--------------

The Downwind and Iterative scripts in the repository's ``examples`` directory
accept ``-o/--outputs`` followed by one or more names from ``ModelBook.outputs``.
The default is ``farm_results``. For example, from ``examples/timeseries``:

.. code-block:: console

    uv run run.py -nf -o farm_results farm_results_write_n1000
    uv run run.py -nf -o create_gif

Writer-only selections complete the calculation and write files without
requiring an in-memory farm dataset for subsequent reporting or plots.
Composed selections retain reporting when a collector returns ``farm_results``.
Use ``farm_results_all`` instead of ``farm_results`` when later point evaluation
or custom reports need fields outside the default set. ``-nf`` suppresses
interactive figures, not explicitly requested output files.

Comparison examples apply the selection to every farm calculation, including
optional benchmarks. Registered writers use their configured filenames; repeated
cases can overwrite those files. Configure distinct output filenames in a custom
script when every case must be retained. The power-mask and sector-management
examples retain their example-specific ``farm_results`` variable selections.
The Gaussian partial-wake comparison saves its optional PNG with
``--figure-file``, separately from ``-o/--outputs``.

Sequential iterator examples, YAML-only examples and the standalone wake-field
exporter do not use this script option. YAML output configuration remains in its
parameter file.

Notebooks
---------

The following examples are jupyter notebooks,
available in the `notebooks <https://github.com/FraunhoferIWES/foxes/tree/main/notebooks>`_
folder of the `foxes <https://github.com/FraunhoferIWES/foxes>`_ github repository.

    .. toctree::
        :titlesonly:

        notebooks/model_book
        notebooks/rotor_models
        notebooks/single_row
        notebooks/timeseries
        notebooks/multi_height
        notebooks/wind_rose
        notebooks/sector_man
        notebooks/heterogeneous
        notebooks/power_mask
        notebooks/yawed_wake
        notebooks/dyn_wakes
        notebooks/timelines
        notebooks/sequential
        notebooks/blockage_farm
        notebooks/blockage_comparison
        notebooks/operating_flag
        notebooks/partial_wakes
        notebooks/icon_dream.ipynb
        notebooks/turbine_data.ipynb
