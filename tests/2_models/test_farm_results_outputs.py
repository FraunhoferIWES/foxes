"""Regression tests for chunk-local farm outputs and their unified results."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.core import FData, MData, Output, OutputList
from foxes.models.outputs import (
    FarmResultsChunkWriter,
    FarmResultsCountWriter,
    FarmResultsOutput,
    FarmResultsPeriodWriter,
)


@pytest.fixture
def output_algo():
    """Provide synthetic farm metadata without running wake models."""
    # Supply distinct metadata for each turbine.
    farm = SimpleNamespace(
        turbine_names=["T0", "T1"],
        wind_farm_list=["north", "south"],
        cluster_list=["A", "B"],
    )
    return SimpleNamespace(farm=farm)


def _chunk(output, algo, labels, start, index, n_chunks, weights=None, point_index=0):
    """Calculate one synthetic chunk with recognizable state/turbine values."""
    # Build valid chunk containers and stable global values.
    n_states = len(labels)
    values = np.arange(start * 2, (start + n_states) * 2, dtype=float).reshape(-1, 2)
    mdata = MData(
        data={FC.STATE: np.asarray(labels)},
        dims={FC.STATE: (FC.STATE,)},
        states_i0=start,
        chunki_states=index,
        chunki_points=point_index,
        n_chunks_states=n_chunks,
        n_chunks_points=1,
    )
    fdata = FData(
        data={
            FC.STATE: np.asarray(labels),
            FC.TURBINE: np.arange(2),
            FV.P: values,
            FV.AMB_P: values + 10.0,
            FV.ORDER: np.tile([1, 0], (n_states, 1)),
            FV.WEIGHT: np.ones_like(values) if weights is None else weights,
        },
        dims={
            FC.STATE: (FC.STATE,),
            FC.TURBINE: (FC.TURBINE,),
            **{
                v: (FC.STATE, FC.TURBINE) for v in [FV.P, FV.AMB_P, FV.ORDER, FV.WEIGHT]
            },
        },
    )

    # Execute the same output hook used by engine runners.
    return output.calculate(algo, mdata, fdata, None, {})


@pytest.mark.parametrize("point_index", [None, 0, 1])
def test_farm_collection_uses_only_first_or_unindexed_point_chunk(
    output_algo, point_index
):
    output = FarmResultsOutput(variables=[FV.P])
    results = _chunk(output, output_algo, [0, 1], 0, 0, 1, point_index=point_index)
    assert (FC.FARM_RESULTS in results) == (point_index in (None, 0))


def _run(output, algo, labels, chunk_sizes, batched=False):
    """Feed chunks incrementally or in one ordered engine update."""
    # Accumulate result-name keyed payloads as the pool engine does.
    results = foxes.utils.Dict()
    pending = foxes.utils.Dict()
    start = 0
    for index, size in enumerate(chunk_sizes):
        chunk = _chunk(
            output, algo, labels[start : start + size], start, index, len(chunk_sizes)
        )
        for name, payload in chunk.items():
            pending.setdefault(name, {})[(index, 0)] = payload
        if not batched:
            output.update_results(algo, len(chunk_sizes), 1, pending, results)
        start += size

    # Finalize only after all state chunks have been delivered.
    if batched:
        output.update_results(algo, len(chunk_sizes), 1, pending, results)
    output.finalize_results(algo, len(chunk_sizes), 1, results)
    return results


def _read_files(directory):
    """Load written datasets while closing every NetCDF handle."""
    # Copy file data before closing the backing handles.
    datasets = []
    for path in sorted(directory.glob("*.nc")):
        with xr.open_dataset(path) as dataset:
            datasets.append(dataset.load())
    return datasets


@pytest.mark.parametrize("chunk_sizes", [[7], [2, 3, 2], [1] * 7])
@pytest.mark.parametrize("batched", [False, True])
def test_farm_results_preserve_data_and_metadata(
    output_algo, tmp_path, chunk_sizes, batched
):
    # Preserve non-default state labels through every chunk layout.
    labels = np.array([0, 1, 5, 6, 8, 9, 10])
    output = FarmResultsOutput(out_dir=tmp_path)
    results = _run(output, output_algo, labels, chunk_sizes, batched)
    dataset = results[FC.FARM_RESULTS]

    # Check values, dimensions, coordinates and farm metadata.
    np.testing.assert_array_equal(dataset[FC.STATE], labels)
    np.testing.assert_array_equal(dataset[FV.P], np.arange(14).reshape(7, 2))
    np.testing.assert_array_equal(dataset[FC.TURBINE], [0, 1])
    assert dataset[FV.P].dims == (FC.STATE, FC.TURBINE)
    assert dataset[FV.WEIGHT].dims == (FC.STATE,)
    assert dataset[FV.ORDER].dtype == np.dtype(foxes.config.dtype_int)
    assert dataset[FC.TNAME].values.tolist() == ["T0", "T1"]
    assert dataset[FC.FARM].values.tolist() == ["north", "south"]
    assert dataset[FC.CLUSTER].values.tolist() == ["A", "B"]


@pytest.mark.parametrize(
    "ambient,waked,expected",
    [(True, False, [FV.AMB_P]), (False, True, [FV.P]), (False, False, [])],
)
def test_farm_results_filter_variables(output_algo, tmp_path, ambient, waked, expected):
    # Select only the requested ambient/waked physical fields.
    output = FarmResultsOutput(
        variables=[FV.P, FV.AMB_P],
        ambient_vars=ambient,
        waked_vars=waked,
        add_turbine_names=False,
        add_farm_names=False,
        add_cluster_names=False,
        out_dir=tmp_path,
    )
    dataset = _run(output, output_algo, np.arange(3), [1, 2])[FC.FARM_RESULTS]

    # Keep dimensions even when no physical variables are requested.
    assert set(dataset.data_vars) == set(expected)
    assert dict(dataset.sizes) == {FC.STATE: 3, FC.TURBINE: 2}


def test_farm_results_weights_can_vary_between_turbines(output_algo, tmp_path):
    # Mix uniform and turbine-specific weights across chunk boundaries.
    output = FarmResultsOutput(out_dir=tmp_path)
    results = foxes.utils.Dict()
    for index, weights in enumerate([np.ones((2, 2)), np.array([[1.0, 2.0]])]):
        chunk = _chunk(
            output,
            output_algo,
            np.arange(index * 2, index * 2 + len(weights)),
            index * 2,
            index,
            2,
            weights,
        )
        pending = foxes.utils.Dict(
            {name: {(index, 0): payload} for name, payload in chunk.items()}
        )
        output.update_results(output_algo, 2, 1, pending, results)
    output.finalize_results(output_algo, 2, 1, results)

    # Only squeeze weights when all unified states are turbine-independent.
    weights = results[FC.FARM_RESULTS][FV.WEIGHT]
    assert weights.dims == (FC.STATE, FC.TURBINE)
    np.testing.assert_array_equal(weights, [[1, 1], [1, 1], [1, 2]])


@pytest.mark.parametrize("chunk_sizes", [[9], [2, 3, 4], [1] * 9])
@pytest.mark.parametrize("batched", [False, True])
def test_count_writer_enforces_limit_on_every_file(
    output_algo, tmp_path, chunk_sizes, batched
):
    # Write an exact limit plus a final short file regardless of engine chunks.
    output = FarmResultsCountWriter(
        "farm",
        2,
        out_dir=tmp_path,
        write_nc_pars={"nc_engine": "h5netcdf", "pack": False, "verbosity": 0},
    )
    with foxes.Engine.new("numpy", verbosity=0):
        results = _run(output, output_algo, np.arange(9), chunk_sizes, batched)
    datasets = _read_files(tmp_path)

    # Verify the required file count, thresholds and global state order.
    assert results == {}
    assert [dataset.sizes[FC.STATE] for dataset in datasets] == [2, 2, 2, 2, 1]
    np.testing.assert_array_equal(
        np.concatenate([ds[FC.STATE] for ds in datasets]), np.arange(9)
    )
    np.testing.assert_array_equal(
        np.concatenate([ds[FV.P] for ds in datasets]), np.arange(18).reshape(9, 2)
    )
    assert [ds.attrs["index"] for ds in datasets] == list(range(5))
    assert all(ds.attrs["n_states"] == ds.sizes[FC.STATE] for ds in datasets)
    assert all(ds.attrs["n_states_per_file"] == 2 for ds in datasets)
    assert all(ds[FV.P].encoding["dtype"] == np.dtype("float64") for ds in datasets)


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True])
def test_count_writer_rejects_invalid_limit(tmp_path, invalid):
    # Reject invalid limits rather than looping forever or writing one giant file.
    with pytest.raises(ValueError, match="n_states_per_file"):
        FarmResultsCountWriter("farm", invalid, out_dir=tmp_path)


@pytest.mark.parametrize(
    "period,labels,names,sizes",
    [
        (
            "day",
            ["2026-01-01T23", "2026-01-02T00", "2026-01-02T01", "2026-01-04T00"],
            ["farm_2026-01-01.nc", "farm_2026-01-02.nc", "farm_2026-01-04.nc"],
            [1, 2, 1],
        ),
        (
            "week",
            ["2025-12-28", "2025-12-29", "2026-01-04", "2026-01-05"],
            ["farm_2025-W52.nc", "farm_2026-W01.nc", "farm_2026-W02.nc"],
            [1, 2, 1],
        ),
        (
            "month",
            ["2025-12-31", "2026-01-01", "2026-01-31", "2026-03-01"],
            ["farm_2025-12.nc", "farm_2026-01.nc", "farm_2026-03.nc"],
            [1, 2, 1],
        ),
        (
            "year",
            ["2024-12-31", "2025-01-01", "2025-12-31", "2027-01-01"],
            ["farm_2024.nc", "farm_2025.nc", "farm_2027.nc"],
            [1, 2, 1],
        ),
    ],
)
@pytest.mark.parametrize("chunk_sizes", [[4], [1, 2, 1], [1] * 4])
def test_period_writer_uses_calendar_boundaries(
    output_algo, tmp_path, period, labels, names, sizes, chunk_sizes
):
    # Exercise boundaries, missing periods and multiple periods in the final chunk.
    labels = np.asarray(labels, dtype="datetime64[ns]")
    output = FarmResultsPeriodWriter(
        base_name="farm",
        period=period,
        out_dir=tmp_path,
        write_nc_pars={"verbosity": 0},
    )
    with foxes.Engine.new("numpy", verbosity=0):
        _run(output, output_algo, labels, chunk_sizes)
    datasets = _read_files(tmp_path)

    # Match filenames to real calendar periods without empty or overwritten files.
    assert [path.name for path in sorted(tmp_path.glob("*.nc"))] == names
    assert [ds.sizes[FC.STATE] for ds in datasets] == sizes
    np.testing.assert_array_equal(
        np.concatenate([ds[FC.STATE] for ds in datasets]), labels
    )


@pytest.mark.parametrize(
    "labels", [[0, 1], ["2026-01-02", "2026-01-01"], ["2026-01-01", "NaT"]]
)
def test_period_writer_rejects_invalid_states(output_algo, tmp_path, labels):
    # Require real chronological datetime coordinates.
    labels = np.asarray(
        labels, dtype="datetime64[ns]" if isinstance(labels[0], str) else int
    )
    output = FarmResultsPeriodWriter(base_name="farm", period="day", out_dir=tmp_path)
    with foxes.Engine.new("numpy", verbosity=0):
        with pytest.raises(ValueError, match="datetime|chronological|NaT"):
            _run(output, output_algo, labels, [2])


def test_period_writer_rejects_invalid_period(tmp_path):
    # Validate the public calendar-period selector at construction.
    with pytest.raises(ValueError, match="period"):
        FarmResultsPeriodWriter(base_name="farm", period="hour", out_dir=tmp_path)


@pytest.mark.parametrize("writer_first", [False, True])
def test_outputs_compose_without_consuming_each_others_results(
    output_algo, tmp_path, writer_first
):
    # Use different selections to expose accidental shared-result reuse.
    dataset_output = FarmResultsOutput(variables=[FV.P], out_dir=tmp_path)
    writer = FarmResultsCountWriter(
        "ambient",
        2,
        variables=[FV.AMB_P],
        out_dir=tmp_path,
        write_nc_pars={"verbosity": 0},
    )
    outputs = [writer, dataset_output] if writer_first else [dataset_output, writer]
    output = OutputList(outputs=outputs, out_dir=tmp_path)
    with foxes.Engine.new("numpy", verbosity=0):
        results = _run(output, output_algo, np.arange(5), [2, 3])

    # Both outputs must retain their own physical variable selection.
    assert FV.P in results[FC.FARM_RESULTS] and FV.AMB_P not in results[FC.FARM_RESULTS]
    datasets = _read_files(tmp_path)
    assert len(datasets) == 3
    assert all(FV.AMB_P in ds and FV.P not in ds for ds in datasets)


def test_chunk_writer_preserves_shared_results_and_file_options(output_algo, tmp_path):
    # Compose a chunk writer with an in-memory dataset output.
    writer = FarmResultsChunkWriter(
        "chunks",
        variables=[FV.AMB_P],
        out_dir=tmp_path,
        out_fname_fun=lambda path: Path("renamed_" + path.name),
        write_nc_pars={"nc_engine": "h5netcdf", "pack": False, "verbosity": 0},
    )
    output = OutputList(
        outputs=[FarmResultsOutput(variables=[FV.P], out_dir=tmp_path), writer],
        out_dir=tmp_path,
    )
    results = _run(output, output_algo, np.arange(5), [2, 3])
    datasets = _read_files(tmp_path)

    # Check per-chunk filenames, metadata and independent output variables.
    assert [p.name for p in sorted(tmp_path.glob("*.nc"))] == [
        "renamed_chunks_000000.nc",
        "renamed_chunks_000001.nc",
    ]
    assert [ds.sizes[FC.STATE] for ds in datasets] == [2, 3]
    assert [ds.attrs["chunki_states"] for ds in datasets] == [0, 1]
    assert all(ds.attrs["n_chunks_states"] == 2 for ds in datasets)
    assert all(FV.AMB_P in ds and FV.P not in ds for ds in datasets)
    assert FV.P in results[FC.FARM_RESULTS]


def test_output_lists_do_not_share_mutable_defaults(tmp_path):
    # Mutating one newly created list must not affect another.
    first = OutputList(out_dir=tmp_path)
    second = OutputList(out_dir=tmp_path)
    first.add_output(FarmResultsOutput(out_dir=tmp_path))
    assert len(first.outputs) == 1
    assert second.outputs == []


@pytest.mark.parametrize("engine", ["numpy", "threads"])
def test_downwind_accepts_multiple_output_instances(tmp_path, engine):
    # Execute real farm model preparation and engine output hooks on synthetic inflow.
    farm = foxes.WindFarm()
    for xy in [[0.0, 0.0], [600.0, 0.0]]:
        farm.add_turbine(foxes.Turbine(xy=xy, turbine_models=["NREL5MW"]), verbosity=0)
    states = foxes.input.states.SingleStateStates(ws=8.0, wd=270.0, ti=0.08, rho=1.225)
    algo = foxes.algorithms.Downwind(farm, states, wake_models=[], verbosity=0)
    outputs = [
        FarmResultsOutput(variables=[FV.P], out_dir=tmp_path),
        FarmResultsCountWriter(
            "farm",
            1,
            variables=[FV.P],
            out_dir=tmp_path,
            write_nc_pars={"nc_engine": "h5netcdf", "verbosity": 0},
        ),
    ]
    with foxes.Engine.new(engine, n_procs=2, verbosity=0):
        results = algo.calc_farm(outputs=outputs)

    # Preserve the multi-output mapping even when only one output returns a dataset.
    assert isinstance(results, dict)
    assert isinstance(results[FC.FARM_RESULTS], xr.Dataset)
    xr.testing.assert_allclose(
        results[FC.FARM_RESULTS][FV.P], _read_files(tmp_path)[0][FV.P]
    )


def test_model_book_registers_only_core_based_farm_outputs():
    # Verify named outputs and count-writer factories share the new core contract.
    book = foxes.ModelBook()
    for name in [
        "farm_results",
        "farm_results_amb",
        "farm_results_write_chunks",
        "farm_results_write_n2",
        "farm_results_write_day",
        "farm_results_write_week",
        "farm_results_write_month",
        "farm_results_write_year",
    ]:
        assert isinstance(book.outputs[name], Output)


@pytest.mark.parametrize(
    "variable,dims,error",
    [("missing", None, KeyError), (FV.P, (FC.STATE,), ValueError)],
)
def test_farm_results_reports_invalid_selected_data(
    output_algo, tmp_path, variable, dims, error
):
    # Construct an unavailable variable or deliberately malformed farm field.
    output = FarmResultsOutput(variables=[variable], out_dir=tmp_path)
    mdata = MData(data={FC.STATE: np.arange(2)}, dims={FC.STATE: (FC.STATE,)})
    fdata = FData(
        data={FC.STATE: np.arange(2), FC.TURBINE: np.arange(2), FV.P: np.ones((2, 2))},
        dims={
            FC.STATE: (FC.STATE,),
            FC.TURBINE: (FC.TURBINE,),
            FV.P: (FC.STATE, FC.TURBINE),
        },
    )
    if dims is not None:
        fdata.dims[FV.P] = dims

    # Surface a contextual exception rather than an optimization-dependent assertion.
    with pytest.raises(error, match=variable):
        output.calculate(output_algo, mdata, fdata, None, {})


def test_farm_results_omit_unassigned_metadata(output_algo, tmp_path):
    # Do not create farm or cluster fields containing only missing memberships.
    output_algo.farm.wind_farm_list = [None, None]
    output_algo.farm.cluster_list = [None, None]
    dataset = _run(FarmResultsOutput(out_dir=tmp_path), output_algo, np.arange(2), [2])[
        FC.FARM_RESULTS
    ]
    assert FC.TNAME in dataset
    assert FC.FARM not in dataset and FC.CLUSTER not in dataset


@pytest.mark.parametrize(
    "writer_type",
    [FarmResultsChunkWriter, FarmResultsCountWriter, FarmResultsPeriodWriter],
)
def test_writers_forward_packing_options(output_algo, tmp_path, writer_type):
    # Require real integer packing so ignored write options cannot pass this test.
    options = dict(
        base_name="packed",
        variables=[FV.P],
        out_dir=tmp_path,
        write_nc_pars={"nc_engine": "h5netcdf", "pack": True, "verbosity": 0},
    )
    if writer_type is FarmResultsCountWriter:
        options["n_states_per_file"] = 2
    elif writer_type is FarmResultsPeriodWriter:
        options["period"] = "day"
    output = writer_type(**options)
    labels = np.array(["2026-01-01", "2026-01-01"], dtype="datetime64[ns]")
    with foxes.Engine.new("numpy", verbosity=0):
        _run(output, output_algo, labels, [2])

    # Inspect the stored representation and verify lossless reconstruction.
    dataset = _read_files(tmp_path)[0]
    assert dataset[FV.P].encoding["dtype"].kind in "iu"
    np.testing.assert_array_equal(dataset[FV.P], [[0, 1], [2, 3]])


def test_count_writer_propagates_worker_write_failure(
    output_algo, tmp_path, monkeypatch
):
    # Simulate an asynchronous worker failure rather than a submission failure.
    output = FarmResultsCountWriter("farm", 2, out_dir=tmp_path)
    with foxes.Engine.new("numpy", verbosity=0) as engine:
        monkeypatch.setattr(engine, "submit", lambda *args, **kwargs: {"failed": True})

        def fail_write(future):
            assert future == {"failed": True}
            raise OSError("synthetic write failure")

        monkeypatch.setattr(engine, "await_result", fail_write)
        with pytest.raises(OSError, match="synthetic write failure"):
            _run(output, output_algo, np.arange(3), [3])


@pytest.mark.parametrize("period", ["day", "week", "month", "year"])
def test_period_writer_handles_one_sample(output_algo, tmp_path, period):
    # Flush the only sample without relying on a previously observed boundary.
    output = FarmResultsPeriodWriter(
        "farm", period=period, out_dir=tmp_path, write_nc_pars={"verbosity": 0}
    )
    with foxes.Engine.new("numpy", verbosity=0):
        _run(output, output_algo, np.array(["2026-01-01"], dtype="datetime64[ns]"), [1])
    datasets = _read_files(tmp_path)
    assert len(datasets) == 1
    assert datasets[0].sizes[FC.STATE] == 1


def test_period_writer_checks_order_after_a_buffer_is_flushed(output_algo, tmp_path):
    # Reject an earlier timestamp even if the previous buffered period was written.
    output = FarmResultsPeriodWriter(
        "farm", period="day", out_dir=tmp_path, write_nc_pars={"verbosity": 0}
    )
    labels = np.array(
        ["2026-01-01", "2026-01-02", "2026-01-01"], dtype="datetime64[ns]"
    )
    with foxes.Engine.new("numpy", verbosity=0):
        with pytest.raises(ValueError, match="chronological"):
            _run(output, output_algo, labels, [2, 1])


@pytest.mark.parametrize("engine_name", ["numpy", "threads"])
@pytest.mark.parametrize("chunk_size", [1, 3, 10])
@pytest.mark.parametrize("normalize", [False, True])
def test_downwind_outputs_are_chunk_independent(
    tmp_path, engine_name, chunk_size, normalize
):
    # Vary inflow directions to exercise downwind-to-farm reordering in real chunks.
    table = pd.DataFrame(
        {FV.WS: [7.0, 8.0, 9.0, 10.0, 11.0], FV.WD: [90.0, 270.0, 90.0, 270.0, 90.0]},
        index=[0, 1, 4, 5, 8],
    )
    states = foxes.input.states.StatesTable(
        table,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
    )
    farm = foxes.WindFarm()
    for xy in [[0.0, 0.0], [600.0, 0.0]]:
        farm.add_turbine(foxes.Turbine(xy=xy, turbine_models=["NREL5MW"]), verbosity=0)
    algo = foxes.algorithms.Downwind(
        farm, states, wake_models=["Jensen_linear_k007"], verbosity=0
    )

    # Compare chunked output with a single-chunk reference using the supported pool path.
    with foxes.Engine.new("numpy", chunk_size_states=10, verbosity=0):
        reference = algo.calc_farm()
    output = FarmResultsOutput(out_dir=tmp_path)
    writers = [
        FarmResultsCountWriter(
            "farm",
            2,
            variables=[FV.P],
            out_dir=tmp_path / "count",
            write_nc_pars={"nc_engine": "h5netcdf", "verbosity": 0},
        ),
        FarmResultsChunkWriter(
            "farm",
            variables=[FV.P],
            out_dir=tmp_path / "chunks",
            write_nc_pars={"nc_engine": "h5netcdf", "verbosity": 0},
        ),
    ]
    with foxes.Engine.new(
        engine_name, n_procs=2, chunk_size_states=chunk_size, verbosity=0
    ):
        results = algo.calc_farm(outputs=[output, *writers], normalize=normalize)[
            FC.FARM_RESULTS
        ]
    for variable in results.data_vars:
        if variable in reference:
            xr.testing.assert_allclose(results[variable], reference[variable])
    np.testing.assert_array_equal(results[FC.STATE], table.index)
    if normalize:
        np.testing.assert_allclose(results[FV.WEIGHT].sum(dim=FC.STATE), 1.0)

    # Check that both writer paths preserve farm order and every state exactly once.
    for folder in ["count", "chunks"]:
        datasets = _read_files(tmp_path / folder)
        np.testing.assert_array_equal(
            np.concatenate([ds[FC.STATE] for ds in datasets]), table.index
        )
        np.testing.assert_allclose(
            np.concatenate([ds[FV.P] for ds in datasets]), reference[FV.P]
        )
    assert [ds.sizes[FC.STATE] for ds in _read_files(tmp_path / "count")] == [2, 2, 1]


@pytest.mark.parametrize("output", [None, OutputList()])
def test_downwind_can_skip_output_collection(tmp_path, output, capsys):
    # A calculation with no output must not dereference a missing output model.
    farm = foxes.WindFarm()
    farm.add_turbine(
        foxes.Turbine(xy=[0.0, 0.0], turbine_models=["NREL5MW"]), verbosity=0
    )
    states = foxes.input.states.SingleStateStates(ws=8.0, wd=270.0, ti=0.08, rho=1.225)
    algo = foxes.algorithms.Downwind(farm, states, wake_models=[], verbosity=0)
    with foxes.Engine.new("numpy", verbosity=0):
        result = algo.calc_farm(outputs=output)
    assert result == {}
    assert capsys.readouterr().out == ""


def test_farm_metadata_lists_preserve_unassigned_memberships():
    # Output metadata needs one entry per turbine, including missing memberships.
    farm = foxes.WindFarm()
    farm.add_turbine(foxes.Turbine(xy=[0.0, 0.0]), verbosity=0)
    farm.add_turbine(
        foxes.Turbine(xy=[1.0, 0.0], wind_farm_name="north", cluster_name="A"),
        verbosity=0,
    )
    assert farm.wind_farm_list == [None, "north"]
    assert farm.cluster_list == [None, "A"]


def test_multiple_buffered_writers_have_independent_results(output_algo, tmp_path):
    # Two count writers must not share private payloads or flush thresholds.
    output = OutputList(
        outputs=[
            FarmResultsCountWriter(
                "power",
                2,
                variables=[FV.P],
                out_dir=tmp_path,
                write_nc_pars={"verbosity": 0},
            ),
            FarmResultsCountWriter(
                "ambient",
                3,
                variables=[FV.AMB_P],
                out_dir=tmp_path,
                write_nc_pars={"verbosity": 0},
            ),
        ],
        out_dir=tmp_path,
    )
    with foxes.Engine.new("numpy", verbosity=0):
        results = _run(output, output_algo, np.arange(7), [2, 1, 4])
    assert results == {}
    assert len(list(tmp_path.glob("power_*.nc"))) == 4
    assert len(list(tmp_path.glob("ambient_*.nc"))) == 3


@pytest.mark.parametrize("all_variables", [False, True])
def test_output_list_merges_and_refreshes_variable_requirements(
    tmp_path, all_variables
):
    # Merge requirements without changing caller-owned lists or losing order.
    variables = [FV.P, FV.AMB_P]
    first = FarmResultsOutput(variables=variables, out_dir=tmp_path)
    second = FarmResultsOutput(
        variables=None if all_variables else [FV.P, FV.WEIGHT], out_dir=tmp_path
    )
    outputs = [first]
    chain = OutputList(outputs=outputs, out_dir=tmp_path)
    chain.add_output(second)
    assert len(outputs) == 1
    assert variables == [FV.P, FV.AMB_P]
    assert chain.mdata_variables == [FC.STATE]
    assert chain.tdata_variables == []
    assert chain.fdata_variables == (
        None if all_variables else [FV.P, FV.AMB_P, FV.WEIGHT]
    )
