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
    uv run run.py -nf -o create_turbine_gif

Writer-only selections complete the calculation and write files without
requiring an in-memory farm dataset for subsequent reporting or plots.
Composed selections retain reporting when a collector returns ``farm_results``.
Use ``farm_results_all`` instead of ``farm_results`` when later point evaluation
or custom reports need fields outside the default set. ``-nf`` suppresses
interactive figures, not explicitly requested output files.

``create_turbine_gif`` and ``create_turbine_mp4`` add a synchronized
turbine-REWS panel to the flow animation. They write ``ani_WS.gif`` and
``ani_WS.mp4``, using the same filenames as the flow-only outputs. Selecting
both variants of the same format overwrites the earlier file unless their
base names are customized. MP4 requires ffmpeg.
Configure the registered instance's ``turbines`` and
``turbine_variable`` before calculation when only selected turbines or another
farm variable should be traced.

Comparison examples apply the selection to every farm calculation, including
optional benchmarks. Registered writers use their configured filenames; repeated
cases can overwrite those files. Configure distinct output filenames in a custom
script when every case must be retained. The power-mask and sector-management
examples retain their example-specific ``farm_results`` variable selections.
The Gaussian partial-wake comparison saves its optional PNG with
``--figure-file``, separately from ``-o/--outputs``.

The ``dyn_wakes`` example locally configures ``create_turbine_gif`` and
``create_turbine_mp4`` using ``FlowTurbineAnimation``. These outputs retain its two-panel flow
and turbine-REWS animation, including layout annotations and the reference
marker. Use ``-o create_turbine_gif`` or ``-o create_turbine_mp4`` to write
``ani_WS.gif`` or ``ani_WS.mp4``; MP4 requires ffmpeg.
``-F/--fps`` sets the frame rate.
Include ``farm_results`` in the selection when result reporting is also wanted.

``timelines`` uses the same turbine-animation selections and retains its
original synchronized flow/REWS panels.
The ``sequential`` and ``sequential_yawcontroller`` scripts accept
``-o create_gif`` or ``-o create_mp4`` for their original flow-only animations,
and optionally ``farm_results`` for reporting. They use
``SeqAnimationOutputPlugin`` to evaluate points during each state step and render
only at iterator completion. Other output writer types are not supported by
these two scripts. ``--debug`` retains wake point/vector overlays; ``-F`` sets
the frame rate. Animation selection replaces their old animation flags.

YAML-only examples and the standalone wake-field exporter do not use this
script option. YAML output configuration remains in its parameter file.

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

    .. toctree::
        :titlesonly:

        notebooks/outputs
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
