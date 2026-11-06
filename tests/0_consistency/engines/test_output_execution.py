"""Backend parity, progress and cleanup for chunk-output execution."""

import io
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.core import MData
from foxes.engines.progress import CalculationProgress
from foxes.models.outputs import (
    FarmResultsOutput,
    FarmResultsChunkWriter,
    FarmResultsCountWriter,
)


def _algorithm(algorithm="Downwind"):
    # Reverse inflow direction and retain non-consecutive state labels.
    table = pd.DataFrame(
        {FV.WS: [7.0, 8.0, 9.0, 10.0], FV.WD: [90.0, 270.0, 90.0, 270.0]},
        index=[1, 3, 5, 9],
    )
    states = foxes.input.states.StatesTable(
        table,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
    )
    farm = foxes.WindFarm()
    for xy in ([0.0, 0.0], [600.0, 0.0]):
        farm.add_turbine(foxes.Turbine(xy=xy, turbine_models=["NREL5MW"]), verbosity=0)
    return getattr(foxes.algorithms, algorithm)(
        farm,
        states,
        rotor_model="centre",
        wake_models=["Jensen_linear_k007"],
        verbosity=0,
    )


@pytest.mark.parametrize(
    "backend",
    [
        "single",
        "numpy",
        "threads",
        "process",
        "multiprocess",
        "dask",
        "local_cluster",
        "ray",
        "mpi",
    ],
)
def test_farm_and_point_output_backend_parity(backend):
    # External runtimes are opt-in so the default suite needs no cluster services.
    if (
        backend in ("ray", "mpi")
        and os.environ.get("FOXES_TEST_CLUSTER_ENGINES") != "1"
    ):
        pytest.skip(
            "Set FOXES_TEST_CLUSTER_ENGINES=1 to test installed Ray/MPI runtimes"
        )

    # Optional engines remain optional in a base test environment.
    module = {
        "multiprocess": "multiprocess",
        "dask": "dask",
        "local_cluster": "distributed",
        "ray": "ray",
        "mpi": "mpi4py",
    }.get(backend)
    if module is not None:
        pytest.importorskip(module)

    # Compare both chunk axes and previous farm data against synchronous execution.
    algo = _algorithm()
    points = np.array([[800.0, 0.0, 90.0], [1000.0, 0.0, 90.0], [1200.0, 0.0, 90.0]])
    with foxes.Engine.new("single", verbosity=0):
        reference = algo.calc_farm(outputs="farm_results_all")
        point_reference = algo.calc_points(reference, points)
    options = (
        {"cluster_pars": {"dashboard_address": None}}
        if backend == "local_cluster"
        else {}
    )
    with foxes.Engine.new(
        backend,
        n_procs=3,
        chunk_size_states=2,
        chunk_size_points=2,
        verbosity=0,
        **options,
    ):
        results = algo.calc_farm(outputs="farm_results_all")
        point_results = algo.calc_points(results, points)
        assert algo.calc_farm(outputs=None) == {}
    xr.testing.assert_allclose(results, reference)
    xr.testing.assert_allclose(point_results, point_reference)


def test_chunk_store_selection_allows_concurrent_chunk_insertion(monkeypatch):
    # Reuse a larger cached state chunk for a smaller calculation.
    algo = _algorithm()
    full = MData(
        {FC.STATE: np.array([1, 3, 5, 9])},
        dims={FC.STATE: (FC.STATE,)},
        states_i0=0,
        chunki_states=0,
        chunki_points=0,
    )
    algo.add_to_chunk_store("cached", np.array([10, 30, 50, 90]), (FC.STATE,), full)
    subset = MData(
        {FC.STATE: np.array([1, 9])},
        dims={FC.STATE: (FC.STATE,)},
        chunki_states=0,
        chunki_points=0,
    )

    # Insert another chunk exactly while the reader matches cached state labels.
    reading, inserted = Event(), Event()
    intersect = np.intersect1d

    def match_states(*args, **kwargs):
        # Release the writer only after cache traversal has started.
        reading.set()
        assert inserted.wait(timeout=5)
        return intersect(*args, **kwargs)

    def add_chunk():
        # New chunks may appear while another worker reads an existing chunk.
        assert reading.wait(timeout=5)
        algo.chunk_store[(1, 0)] = {}
        inserted.set()

    # A stable traversal must retain the selected values without copying arrays.
    monkeypatch.setattr(np, "intersect1d", match_states)
    with ThreadPoolExecutor(max_workers=1) as pool:
        writer = pool.submit(add_chunk)
        values = algo.get_from_chunk_store("cached", subset)
        writer.result(timeout=5)
    np.testing.assert_array_equal(values, [10, 90])


@pytest.mark.parametrize("algorithm", ["Iterative", "Sequential"])
def test_iterative_algorithms_use_explicit_output_models(algorithm):
    # Intermediate downwind data must remain independent of final selections.
    algo = _algorithm(algorithm)
    with foxes.Engine.new("single", verbosity=0):
        if algorithm == "Sequential":
            results = xr.concat(list(algo), dim=FC.STATE)
        else:
            results = algo.calc_farm(outputs=FarmResultsOutput(variables=[FV.P]))
    assert results.sizes[FC.STATE] == 4
    assert FV.P in results
    assert np.isfinite(results[FV.P]).all()


@pytest.mark.parametrize("mode", [True, False, None])
@pytest.mark.parametrize("verbosity", [0, 1])
def test_redirected_progress_is_bounded_and_newline_only(mode, verbosity):
    # Logs get bounded snapshots, never terminal cursor controls.
    stream = io.StringIO()
    progress = CalculationProgress("test", 101, mode, verbosity, stream=stream)
    for _ in range(101):
        progress.update()
    progress.close()
    text = stream.getvalue()
    if mode is None or verbosity == 0:
        assert text == ""
    else:
        assert "\r" not in text and "\x1b" not in text
        assert len(text.splitlines()) <= 12
        assert "Finished 101/101 chunks (100%)" in text


def test_progress_failure_never_reports_success():
    # Finalization failures must remain failures even after every chunk completed.
    stream = io.StringIO()
    progress = CalculationProgress("test", 1, False, 1, stream=stream)
    progress.update()
    progress.close(error=True)
    assert "Failed" in stream.getvalue()
    assert "Finished" not in stream.getvalue()


@pytest.mark.parametrize("notebook", [False, True])
def test_interactive_progress_closes_on_failure(monkeypatch, notebook):
    # Select a bar for either an interactive terminal or a live notebook kernel.
    events = []
    options = {}
    bar = SimpleNamespace(
        update=lambda value: events.append(value),
        set_description=lambda value: events.append(value),
        close=lambda: events.append("closed"),
    )
    monkeypatch.setattr("foxes.engines.progress._in_notebook", lambda: notebook)

    def create_bar(**kwargs):
        # Preserve automatic encoding detection rather than forcing ASCII output.
        options.update(kwargs)
        return bar

    monkeypatch.setattr("tqdm.auto.tqdm", create_bar)
    stream = io.StringIO()
    monkeypatch.setattr(stream, "isatty", lambda: not notebook)
    progress = CalculationProgress("test", 1, True, 1, stream=stream)
    progress.update()
    progress.close(error=True)
    assert events == [1, "test - failed", "closed"]
    assert options["ascii"] is None


def test_nested_engine_rejection_preserves_active_engine():
    # A rejected entry must not allocate a pool or clear another engine's slot.
    with foxes.Engine.new("numpy", verbosity=0) as active:
        with pytest.raises(ValueError, match="already set"):
            with foxes.Engine.new("process", n_procs=2, verbosity=0):
                pytest.fail("Nested engine unexpectedly entered")
        assert foxes.get_engine() is active
    assert foxes.get_engine(error=False) is None


def test_default_farm_field_selection_is_not_expanded():
    # Full wake/point integration must use an explicit output, not change defaults.
    assert foxes.ModelBook().outputs["farm_results"].fdata_variables == [
        FV.X,
        FV.Y,
        FV.H,
        FV.D,
        FV.WD,
        FV.YAW,
        FV.AMB_RHO,
        FV.AMB_REWS,
        FV.REWS,
        FV.AMB_TI,
        FV.TI,
        FV.AMB_P,
        FV.P,
        FV.WEIGHT,
    ]


@pytest.mark.parametrize(
    "backend",
    ["single", "numpy", "threads", "process", "multiprocess", "dask", "local_cluster"],
)
@pytest.mark.parametrize("split", ["chunks", "count"])
def test_output_files_are_complete_before_engine_returns(tmp_path, backend, split):
    # Avoid requiring the aggregate engines extra for the ordinary test suite.
    module = {
        "multiprocess": "multiprocess",
        "dask": "dask",
        "local_cluster": "distributed",
    }.get(backend)
    if module is not None:
        pytest.importorskip(module)

    # Output models own worker writes and parent-submitted buffered file jobs.
    algo = _algorithm()
    options = (
        {"cluster_pars": {"dashboard_address": None}}
        if backend == "local_cluster"
        else {}
    )
    writer = FarmResultsChunkWriter if split == "chunks" else FarmResultsCountWriter
    split_pars = {} if split == "chunks" else {"n_states_per_file": 3}
    output = writer(
        base_name="power",
        variables=[FV.P],
        out_dir=tmp_path,
        write_nc_pars={"nc_engine": "h5netcdf", "verbosity": 0},
        **split_pars,
    )
    with foxes.Engine.new(
        backend,
        n_procs=3,
        chunk_size_states=2,
        verbosity=0,
        **options,
    ):
        assert algo.calc_farm(outputs=output) == {}
        datasets = []
        for path in sorted(tmp_path.glob("power_*.nc")):
            with xr.open_dataset(path) as dataset:
                datasets.append(dataset.load())
        expected_sizes = (
            [3, 1] if split == "count" else [4] if backend == "single" else [2, 2]
        )
        assert [dataset.sizes[FC.STATE] for dataset in datasets] == expected_sizes
        np.testing.assert_array_equal(
            np.concatenate([dataset[FC.STATE] for dataset in datasets]),
            [1, 3, 5, 9],
        )
        assert all(FV.P in dataset for dataset in datasets)


class _FailingOutput(FarmResultsOutput):
    def finalize_results(self, *args, **kwargs):
        # Fail after workers finish, proving success is not reported prematurely.
        raise OSError("synthetic finalization failure")


def test_finalization_failure_closes_progress_and_releases_engine(capsys):
    # Output failures must propagate while the parent cleans up its display.
    algo = _algorithm()
    with pytest.raises(OSError, match="synthetic finalization failure"):
        with foxes.Engine.new("threads", n_procs=3, verbosity=1, progress_bar=False):
            algo.calc_farm(outputs=_FailingOutput())
    text = capsys.readouterr().err
    assert "Failed" in text
    assert "Finished" not in text
    assert "ThreadsEngine [Downwind_calc]" in text
    assert foxes.get_engine(error=False) is None


def test_failed_task_cleanup_awaits_every_pending_worker():
    # Storage cleanup must not race outstanding tasks or mask the first failure.
    from foxes.engines.threads import ThreadsEngine

    events = []
    engine = ThreadsEngine(n_procs=2, verbosity=0)

    def await_task(task):
        events.append(task)
        if task == "bad":
            raise ValueError("secondary error")

    engine.await_result = await_task
    with pytest.warns(RuntimeWarning, match="secondary error"):
        engine._drain_failed_tasks({0: "bad", 1: "good"})
    assert events == ["bad", "good"]


def test_slurm_uses_common_lifecycle_without_starting_a_local_cluster(monkeypatch):
    # Verify scheduler selection with synthetic infrastructure, not a real queue.
    from foxes.engines import dask as module

    events = []

    class Context:
        dashboard_link = "synthetic dashboard"

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *args):
            events.append("exit")

        def scale(self, **kwargs):
            events.append(kwargs)

        def job_script(self):
            return "synthetic job script"

    monkeypatch.setattr(module, "load_dask", lambda: None)
    monkeypatch.setattr(module, "load_distributed", lambda: None)
    monkeypatch.setattr(
        module,
        "_get_dask",
        lambda: SimpleNamespace(
            config=SimpleNamespace(set=lambda *args: Context()),
        ),
    )
    monkeypatch.setattr(
        module,
        "_get_distributed",
        lambda: SimpleNamespace(
            Client=lambda *args, **kwargs: Context(),
        ),
    )
    monkeypatch.setattr(
        module,
        "import_module",
        lambda *args, **kwargs: SimpleNamespace(
            SLURMCluster=lambda **kwargs: Context(),
        ),
    )
    with module.SlurmClusterEngine(cluster_pars={"nodes": 2}, verbosity=0):
        assert events == ["enter", {"jobs": 2}, "enter", "enter"]
    assert events[-3:] == ["exit", "exit", "exit"]


@pytest.mark.parametrize("encoding,ascii_bar", [("utf-8", False), ("ascii", True)])
def test_terminal_bar_adapts_to_stream_encoding(monkeypatch, encoding, ascii_bar):
    # Exercise tqdm's real encoding detection, not just the factory arguments.
    class Terminal(io.StringIO):
        @property
        def encoding(self):
            return encoding

        def isatty(self):
            return True

    monkeypatch.setattr("foxes.engines.progress._in_notebook", lambda: False)
    progress = CalculationProgress("test", 1, True, 1, stream=Terminal())
    try:
        assert progress.bar.ascii is ascii_bar
        progress.update()
        progress.finalizing()
    finally:
        progress.close()
