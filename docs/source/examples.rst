Examples
========

Python scripts
--------------

The `examples folder on GitHub (main branch)
<https://github.com/FraunhoferIWES/foxes/tree/main/examples>`_ contains runnable
Python scripts and parameter-file cases. Each example directory includes a
``README.md`` describing its purpose, required inputs and run commands.

Start with ``quickstart`` for a minimal wind-farm calculation, ``timeseries``
for time-series evaluation, or ``compare_wakes`` for wake-profile comparisons.
Other examples cover rotor and partial-wake models, heterogeneous inflow,
yaw control and time-dependent wakes.

Running a script
^^^^^^^^^^^^^^^^

After :doc:`installing FOXES <installation>`, change into the selected example
directory and check its options. For example, from the repository root:

.. code-block:: console

    cd examples/timeseries
    uv run python run.py --help
    uv run python run.py -e numpy -nf

Input, model and engine options differ between scripts. Some cases require
optional dependencies, prepared external data or a configured cluster; check
their README before running them.

Selecting results and outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``run.py`` scripts share these options:

* ``-R/--results``: Select one or more calculation output models, defaulting to
  ``farm_results``.
* ``-o/--outputs``: Select one or more subsequent output models. Defaults and
  example-specific model settings vary by script.
* ``-nf/--nofig``: Skip plots, animations, image files and image-only point
  calculations. Numerical slices, point data and NetCDF writers remain enabled.
* ``-v/--verbosity``: Use ``0`` for quiet execution, ``1`` for a summary, ``2``
  for returned results (the default), or ``3`` for library progress as well.

Both output selectors accept model-book names, not physical variable names.
See :doc:`Outputs <outputs>` for result models, variable sets and plotting
choices.

For example, from ``examples/timeseries``, collect farm results while also
writing NetCDF files with up to 1,000 states per file:

.. code-block:: console

    uv run python run.py -e numpy -nf -R farm_results farm_results_write_n1000

To retain all available farm fields and create a flow animation:

.. code-block:: console

    uv run python run.py -R farm_results_all -o create_gif

Use ``farm_results_all`` when subsequent outputs need fields outside the
default selection. MP4 animation outputs require an external ffmpeg
installation. Outputs can share filenames, so configure distinct names in the
script when keeping results from several outputs or comparison cases.
The ``sequential`` and ``sequential_yawcontroller`` examples support farm-result
outputs and their animation models, rather than arbitrary point outputs.

YAML and WindIO cases configure outputs in their parameter files instead of
using these script selectors; see :doc:`Input parameter files <parameter_files>`.

Running several examples
^^^^^^^^^^^^^^^^^^^^^^^^

From the repository root, use the example runner to list cases or run a small,
figure-less selection:

.. code-block:: console

    uv run python examples/run_all.py --dry
    uv run python examples/run_all.py --quick --nofig --include quickstart

``--quick`` uses small local inputs and local engines; ``--nofig`` independently
disables image outputs. The runner works in temporary copies and removes
generated files afterward. Run an individual script directly when you want to
retain its output files.

Notebooks
---------

The following examples are jupyter notebooks,
available in the `notebooks <https://github.com/FraunhoferIWES/foxes/tree/main/notebooks>`_
folder of the `foxes <https://github.com/FraunhoferIWES/foxes>`_ github repository.

Start with :doc:`the practical output guide <notebooks/outputs>` for collecting
turbine metrics, weighted farm/cluster summaries, point results, layout/rose/flow
plots, NetCDF writer policies and a small GIF. It uses synthetic conditions and
a packaged turbine curve, reuses calculated results and cleans temporary files.

The dynamic-wake, timelines and sequential notebooks configure the same
model-book animation outputs and display their saved GIF files. Temporary output
directories are cleaned after display. The sequential notebook preserves its
600 ms frame interval using ``fps=5 / 3``. The timelines notebook uses state
and point chunking, like the script.
Dynamic-wake and timelines calculations retain keyed ``farm_results_all`` first,
then create animations through ``calc_outputs`` with those farm results.
The output guide demonstrates stable keyed returns and ``nofig=True`` suppression
before point-grid requests, using results-oriented terminology.

    .. toctree::
        :titlesonly:

        notebooks/model_book
        notebooks/rotor_models
        notebooks/single_row
        notebooks/timeseries
        notebooks/outputs
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
