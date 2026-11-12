Examples
========

Python scripts
--------------

Every Python script in ``examples`` keeps its workflow in ``main(argv=None)``.
Use ``-R/--results`` for calculation result models (default: ``farm_results``)
and ``-o/--outputs`` for subsequent script-specific output models. Both accept
one or more names from ``ModelBook.outputs``. ``-r`` retains its rotor meaning;
reference points use ``--ref_xy``. For example, from ``examples/timeseries``:

.. code-block:: console

    uv run run.py -nf -R farm_results farm_results_write_n1000
    uv run run.py -R farm_results_all -o create_gif
    uv run run.py -R farm_results_all -o create_turbine_gif

The scripts use ``ret_dict=True`` and print every returned key followed by its
value on a new line at verbosity two and three. Levels zero and one suppress
these values. Subsequent outputs receive the calculation mapping's
``farm_results`` entry, or ``None`` if absent, without special dataframe reports.
Outputs with different grids run independently. Use ``farm_results_all`` when
subsequent numerical consumers need fields outside the configured default set.
``-nf`` skips plots, animations, image files and image-only point calculations.
Scripts configure ``Output.nofig``; the output framework handles suppression
before requesting grids, without script-side type filtering.
Explicit numerical slices, point data, NetCDF writers and scientific input
reductions remain enabled.

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
Select calculation results independently through ``-R/--results``.

``timelines`` uses the same turbine-animation selections and retains its
original synchronized flow/REWS panels.
The ``sequential`` and ``sequential_yawcontroller`` scripts accept
``-o create_gif`` or ``-o create_mp4`` for their original flow-only animations,
and ``-R/--results`` for collection and reporting. They use
``SeqAnimationOutputPlugin`` to evaluate points during each state step and render
only at iterator completion. Farm collectors, tables, maps and writers process
the accumulated iterator data through an output-only Downwind processor.
Other point-dependent outputs are not supported by these two scripts.
``--debug`` retains wake point/vector overlays; ``-F`` sets the frame rate.
Animation plugins skip disabled outputs before requesting points under ``-nf``.

The standalone wake-field exporter defaults to the numerical ``wake_field``
output after farm collection. YAML output configuration remains in its parameter
file rather than using these script selectors.

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
