"""Point collector and writer parity with engine-defined target chunks."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.core import FData, MData, OutputList, TData
from foxes.models.outputs import (
    PointResultsChunkWriter,
    PointResultsCountWriter,
    PointResultsOutput,
    PointResultsPeriodWriter,
)


def _chunk(
    labels, states, targets, n_tpoints=2, moving=True, key=(0, 0), counts=(1, 1)
):
    """Create immutable synthetic data with globally identifiable fields."""
    mdata = MData(
        {FC.STATE: np.asarray(labels)[states]},
        dims={FC.STATE: (FC.STATE,)},
        states_i0=int(states[0]),
        chunki_states=key[0],
        chunki_points=key[1],
        n_chunks_states=counts[0],
        n_chunks_points=counts[1],
    )
    shape = (len(states), len(targets), n_tpoints)
    values = np.broadcast_to(
        states[:, None, None] * 100
        + targets[None, :, None] * 10
        + np.arange(n_tpoints),
        shape,
    ).copy()
    points = np.zeros((*shape, 3))
    points[..., 0] = targets[None, :, None] * 20
    points[..., 1] = np.arange(n_tpoints)
    points[..., 2] = 90
    if moving:
        points[..., 0] += states[:, None, None]
    fields = {FV.WS: values, FV.AMB_WS: values + 1000, FV.WEIGHT: np.ones(shape)}
    data = TData.from_tpoints(points, np.full(n_tpoints, 1 / n_tpoints), mdata=mdata)
    data.add(FC.STATE, np.asarray(labels)[states], (FC.STATE,))
    data.add(FC.TARGET, targets + 10, (FC.TARGET,))
    for name, value in fields.items():
        data.add(name, value, (FC.STATE, FC.TARGET, FC.TPOINT))
    for value in data.values():
        value.flags.writeable = False
    return mdata, data, {FV.WS: values + 0.5}


def _run(
    output, labels, state_sizes=(2, 1, 4), target_sizes=(1, 2), batched=False, **kwargs
):
    """Feed all engine tiles, incrementally or in one parent update."""
    algo = SimpleNamespace(verbosity=0)
    results = foxes.utils.Dict()
    pending = foxes.utils.Dict()
    state_edges = np.r_[0, np.cumsum(state_sizes)]
    target_edges = np.r_[0, np.cumsum(target_sizes)]
    for state, (start, stop) in enumerate(zip(state_edges[:-1], state_edges[1:])):
        for target, (first, last) in enumerate(
            zip(target_edges[:-1], target_edges[1:])
        ):
            mdata, tdata, values = _chunk(
                labels,
                np.arange(start, stop),
                np.arange(first, last),
                key=(state, target),
                counts=(len(state_sizes), len(target_sizes)),
                **kwargs,
            )
            calculated = output.calculate(algo, mdata, FData(), tdata, values)
            for name, payload in calculated.items():
                pending.setdefault(name, {})[(state, target)] = payload
            if not batched:
                output.update_results(
                    algo, len(state_sizes), len(target_sizes), pending, results
                )
    if batched:
        output.update_results(
            algo, len(state_sizes), len(target_sizes), pending, results
        )
    output.finalize_results(algo, len(state_sizes), len(target_sizes), results)
    return results


def _read_files(directory, prefix="points"):
    """Load real NetCDF files and close their backing handles."""
    datasets = []
    for path in sorted(directory.glob(f"{prefix}_*.nc")):
        with xr.open_dataset(path) as dataset:
            datasets.append(dataset.load())
    return datasets


@pytest.mark.parametrize("state_sizes", [(7,), (2, 1, 4), (1,) * 7])
@pytest.mark.parametrize("target_sizes", [(3,), (1, 2)])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("moving", [False, True])
def test_count_writer_spans_both_chunk_axes(
    tmp_path, state_sizes, target_sizes, batched, moving
):
    writer = PointResultsCountWriter(
        "points", 3, variables=[FV.WS], out_dir=tmp_path, write_nc_pars={"verbosity": 0}
    )
    with foxes.Engine.new("numpy", verbosity=0):
        assert (
            _run(
                writer,
                np.arange(7) * 2,
                state_sizes,
                target_sizes,
                batched,
                moving=moving,
            )
            == {}
        )
    datasets = _read_files(tmp_path)
    assert [ds.sizes[FC.STATE] for ds in datasets] == [3, 3, 1]
    assert [ds.attrs["index"] for ds in datasets] == [0, 1, 2]
    assert all(ds.attrs["n_states_per_file"] == 3 for ds in datasets)
    assert all(ds.sizes[FC.TARGET] == 3 and ds.sizes[FC.TPOINT] == 2 for ds in datasets)
    combined = xr.concat(datasets, FC.STATE)
    expected = (
        np.arange(7)[:, None, None] * 100
        + np.arange(3)[None, :, None] * 10
        + np.arange(2)
        + 0.5
    )
    np.testing.assert_array_equal(combined[FV.WS], expected)
    np.testing.assert_array_equal(combined[FC.STATE], np.arange(7) * 2)
    np.testing.assert_array_equal(combined[FC.TARGET], [10, 11, 12])
    expected_x = np.broadcast_to(
        np.arange(3)[None, :, None] * 20, expected.shape
    ).copy()
    if moving:
        expected_x += np.arange(7)[:, None, None]
    np.testing.assert_array_equal(combined[FV.X], expected_x)
    assert FV.AMB_WS not in combined


@pytest.mark.parametrize("reverse", [False, True])
def test_composed_collectors_and_writers_select_independent_fields(tmp_path, reverse):
    outputs = [
        PointResultsOutput(variables=[FV.WS], out_dir=tmp_path),
        PointResultsCountWriter(
            "points",
            3,
            variables=[FV.AMB_WS],
            out_dir=tmp_path,
            write_nc_pars={"verbosity": 0},
        ),
        PointResultsCountWriter(
            "weights",
            2,
            variables=[FV.WEIGHT],
            out_dir=tmp_path,
            write_nc_pars={"verbosity": 0},
        ),
    ]
    chain = OutputList(outputs=outputs[::-1] if reverse else outputs, out_dir=tmp_path)
    with foxes.Engine.new("numpy", verbosity=0):
        result = _run(chain, np.arange(7))
    assert set(result) == {FC.POINT_RESULTS}
    assert set(result[FC.POINT_RESULTS].data_vars) == {FV.WS}
    assert all(set(ds.data_vars) == {FV.AMB_WS} for ds in _read_files(tmp_path))
    assert len(_read_files(tmp_path, "weights")) == 4


@pytest.mark.parametrize(
    "ambient,waked,expected",
    [
        (True, True, {FV.WS, FV.AMB_WS}),
        (False, True, {FV.WS}),
        (True, False, {FV.AMB_WS}),
        (False, False, set()),
    ],
)
def test_collector_filters_ambient_and_waked_fields(tmp_path, ambient, waked, expected):
    output = PointResultsOutput(
        variables=[FV.WS, FV.AMB_WS],
        ambient_vars=ambient,
        waked_vars=waked,
        out_dir=tmp_path,
    )
    result = _run(output, np.arange(7))[FC.POINT_RESULTS]
    assert set(result.data_vars) == expected
    assert dict(result.sizes) == {FC.STATE: 7, FC.TARGET: 3, FC.TPOINT: 2}


def test_collector_preserves_string_state_labels(tmp_path):
    output = PointResultsOutput(variables=[], out_dir=tmp_path)
    labels = np.array([f"state-{index}" for index in range(7)])
    result = _run(output, labels)[FC.POINT_RESULTS]
    np.testing.assert_array_equal(result[FC.STATE], labels)
    assert not result.data_vars


def test_chunk_writer_uses_exact_tdata_tiles_and_file_options(tmp_path):
    writer = PointResultsChunkWriter(
        "points",
        variables=[FV.WS],
        out_dir=tmp_path,
        out_fname_fun=lambda path: Path("renamed_" + path.name),
        write_nc_pars={"nc_engine": "h5netcdf", "pack": False, "verbosity": 0},
    )
    assert _run(writer, np.arange(7)) == {}
    files = sorted(tmp_path.glob("*.nc"))
    assert [path.name for path in files] == [
        f"renamed_points_{state:06d}_{target:06d}.nc"
        for state in range(3)
        for target in range(2)
    ]
    for path in files:
        with xr.open_dataset(path) as dataset:
            state = dataset.attrs["chunki_states"]
            target = dataset.attrs["chunki_points"]
            assert dataset.sizes[FC.STATE] == [2, 1, 4][state]
            assert dataset.sizes[FC.TARGET] == [1, 2][target]
            assert dataset.sizes[FC.TPOINT] == 2
            assert dataset.attrs["n_chunks_states"] == 3
            assert dataset.attrs["n_chunks_points"] == 2
            assert dataset[FV.WS].encoding["dtype"].kind == "f"


@pytest.mark.parametrize(
    "period,labels,suffixes,sizes",
    [
        (
            "day",
            ["2021-01-01", "2021-01-01", "2021-01-03", "2021-01-03"],
            ["2021-01-01", "2021-01-03"],
            [2, 2],
        ),
        (
            "week",
            ["2020-12-31", "2021-01-01", "2021-01-04", "2021-01-11"],
            ["2020-W53", "2021-W01", "2021-W02"],
            [2, 1, 1],
        ),
        (
            "month",
            ["2020-12-31", "2021-01-01", "2021-01-31", "2021-03-01"],
            ["2020-12", "2021-01", "2021-03"],
            [1, 2, 1],
        ),
        (
            "year",
            ["2020-12-31", "2021-01-01", "2021-12-31", "2023-01-01"],
            ["2020", "2021", "2023"],
            [1, 2, 1],
        ),
    ],
)
@pytest.mark.parametrize("state_sizes", [(4,), (1, 2, 1)])
def test_period_writers_use_shared_calendar_boundaries(
    tmp_path, period, labels, suffixes, sizes, state_sizes
):
    writer = PointResultsPeriodWriter(
        "points", period=period, out_dir=tmp_path, write_nc_pars={"verbosity": 0}
    )
    with foxes.Engine.new("numpy", verbosity=0):
        assert _run(writer, np.array(labels, dtype="datetime64[ns]"), state_sizes) == {}
    assert [path.name for path in sorted(tmp_path.glob("*.nc"))] == [
        f"points_{suffix}.nc" for suffix in suffixes
    ]
    datasets = _read_files(tmp_path)
    assert [ds.sizes[FC.STATE] for ds in datasets] == sizes
    assert all(
        ds.sizes[FC.TARGET] == 3 and ds.attrs["period"] == period for ds in datasets
    )
    if period == "week":
        assert datasets[0].attrs["iso_year"] == 2020
        assert datasets[0].attrs["week"] == 53


@pytest.mark.parametrize(
    "writer_type",
    [PointResultsChunkWriter, PointResultsCountWriter, PointResultsPeriodWriter],
)
def test_writers_forward_integer_packing(tmp_path, writer_type):
    options = dict(
        base_name="points",
        variables=[FV.AMB_WS],
        out_dir=tmp_path,
        write_nc_pars={"nc_engine": "h5netcdf", "pack": True, "verbosity": 0},
    )
    if writer_type is PointResultsCountWriter:
        options["n_states_per_file"] = 2
    elif writer_type is PointResultsPeriodWriter:
        options["period"] = "day"
    with foxes.Engine.new("numpy", verbosity=0):
        _run(
            writer_type(**options),
            np.array(["2021-01-01"], dtype="datetime64[ns]"),
            (1,),
            (3,),
        )
    dataset = _read_files(tmp_path)[0]
    assert dataset[FV.AMB_WS].encoding["dtype"].kind in "iu"
    np.testing.assert_array_equal(
        dataset[FV.AMB_WS], np.arange(3)[None, :, None] * 10 + np.arange(2) + 1000
    )


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_count_writer_rejects_invalid_limits(tmp_path, limit):
    with pytest.raises(ValueError, match="positive integer"):
        PointResultsCountWriter("points", limit, out_dir=tmp_path)


def test_period_writer_rejects_invalid_period(tmp_path):
    with pytest.raises(ValueError, match="Invalid period"):
        PointResultsPeriodWriter("points", period="hour", out_dir=tmp_path)


@pytest.mark.parametrize(
    "labels,match",
    [
        (np.arange(3), "datetime64"),
        (np.array(["2021-01-01", "NaT", "2021-01-02"], dtype="datetime64[ns]"), "NaT"),
        (
            np.array(
                ["2021-01-02", "2021-01-01", "2021-01-03"], dtype="datetime64[ns]"
            ),
            "chronological",
        ),
        (
            np.array(
                ["2021-01-01", "2021-01-02", "2021-01-01"], dtype="datetime64[ns]"
            ),
            "chronological",
        ),
    ],
)
def test_period_writer_validates_coordinates_across_flushed_chunks(
    tmp_path, labels, match
):
    writer = PointResultsPeriodWriter(
        "points", period="day", out_dir=tmp_path, write_nc_pars={"verbosity": 0}
    )
    with foxes.Engine.new("numpy", verbosity=0):
        with pytest.raises(ValueError, match=match):
            _run(writer, labels, (2, 1))


@pytest.mark.parametrize(
    "output_type,options",
    [
        (PointResultsOutput, {}),
        (PointResultsCountWriter, {"base_name": "points", "n_states_per_file": 2}),
        (PointResultsChunkWriter, {"base_name": "points"}),
    ],
)
def test_point_outputs_reject_missing_tdata(tmp_path, output_type, options):
    output = output_type(out_dir=tmp_path, **options)
    with pytest.raises(ValueError, match="requires target data"):
        output.calculate(SimpleNamespace(verbosity=0), MData(), FData(), None, {})


def test_collector_reports_missing_field(tmp_path):
    output = PointResultsOutput(variables=["missing"], out_dir=tmp_path)
    mdata, tdata, values = _chunk(np.arange(2), np.arange(2), np.arange(2))
    with pytest.raises(KeyError, match="missing"):
        output.calculate(SimpleNamespace(verbosity=0), mdata, FData(), tdata, values)


@pytest.mark.parametrize("key", [(0, 0), (0, 2), (-1, 0), (2, 0)])
def test_buffered_writer_rejects_duplicate_or_invalid_tiles(tmp_path, key):
    output = PointResultsCountWriter("points", 2, out_dir=tmp_path)
    algo = SimpleNamespace(verbosity=0)
    mdata, tdata, values = _chunk(np.arange(2), np.arange(2), np.arange(1))
    calculated = output.calculate(algo, mdata, FData(), tdata, values)
    name, payload = next(iter(calculated.items()))
    results = foxes.utils.Dict()
    output.update_results(
        algo, 1, 2, foxes.utils.Dict({name: {(0, 0): payload}}), results
    )
    with pytest.raises(ValueError, match="Unexpected point chunk"):
        output.update_results(
            algo, 1, 2, foxes.utils.Dict({name: {key: payload}}), results
        )


def test_buffered_writer_rejects_missing_target_tile(tmp_path):
    output = PointResultsCountWriter("points", 2, out_dir=tmp_path)
    algo = SimpleNamespace(verbosity=0)
    mdata, tdata, values = _chunk(np.arange(2), np.arange(2), np.arange(1))
    name, payload = next(
        iter(output.calculate(algo, mdata, FData(), tdata, values).items())
    )
    results = foxes.utils.Dict()
    output.update_results(
        algo, 1, 2, foxes.utils.Dict({name: {(0, 0): payload}}), results
    )
    with pytest.raises(ValueError, match="Missing point result chunks"):
        output.finalize_results(algo, 1, 2, results)


@pytest.mark.parametrize("field", ["states", "targets", "tpoints", "dims"])
def test_collector_rejects_incompatible_tiles(tmp_path, field):
    output = PointResultsOutput(out_dir=tmp_path)
    mdata, tdata, values = _chunk(np.arange(2), np.arange(2), np.arange(1))
    payload = output.calculate(
        SimpleNamespace(verbosity=0), mdata, FData(), tdata, values
    )[FC.POINT_RESULTS]
    changed = payload.copy()
    if field == "dims":
        changed[field] = {}
    else:
        changed[field] = payload[field] + 1
    key = (0, 1) if field == "states" else (1, 0)
    with pytest.raises(ValueError, match="differ between chunks"):
        output._as_result_data({(0, 0): payload, key: changed}, key[0] + 1, key[1] + 1)


@pytest.mark.parametrize(
    "writer_type", [PointResultsCountWriter, PointResultsPeriodWriter]
)
def test_buffered_writer_flushes_single_state(tmp_path, writer_type):
    options = (
        {"n_states_per_file": 3}
        if writer_type is PointResultsCountWriter
        else {"period": "week"}
    )
    writer = writer_type(
        "points", out_dir=tmp_path, write_nc_pars={"verbosity": 0}, **options
    )
    with foxes.Engine.new("numpy", verbosity=0):
        assert (
            _run(
                writer,
                np.array(["2021-01-01"], dtype="datetime64[ns]"),
                (1,),
                n_tpoints=1,
            )
            == {}
        )
    dataset = _read_files(tmp_path)[0]
    assert dict(dataset.sizes) == {FC.STATE: 1, FC.TARGET: 3, FC.TPOINT: 1}


@pytest.mark.parametrize(
    "output_verbosity,level", [(None, 0), (None, 2), (0, 2), (3, 0)]
)
def test_point_writer_inherits_or_overrides_write_verbosity(
    tmp_path, monkeypatch, output_verbosity, level
):
    from foxes.models.outputs import point_results_writers

    calls = []

    def write(ds, fpath, **options):
        calls.append(options)

    monkeypatch.setattr(point_results_writers, "write_nc", write)
    options = {} if output_verbosity is None else {"verbosity": output_verbosity}
    writer = PointResultsChunkWriter("points", out_dir=tmp_path, write_nc_pars=options)
    mdata, tdata, values = _chunk(np.arange(2), np.arange(2), np.arange(1))
    assert (
        writer.calculate(
            SimpleNamespace(verbosity=level), mdata, FData(), tdata, values
        )
        == {}
    )
    assert calls == [
        {"verbosity": level if output_verbosity is None else output_verbosity}
    ]


def test_count_writer_propagates_asynchronous_write_errors(tmp_path, monkeypatch):
    output = PointResultsCountWriter("points", 2, out_dir=tmp_path)
    with foxes.Engine.new("numpy", verbosity=0) as engine:
        monkeypatch.setattr(engine, "submit", lambda *args, **kwargs: {"failed": True})

        def fail_write(future):
            assert future == {"failed": True}
            raise OSError("synthetic point write failure")

        monkeypatch.setattr(engine, "await_result", fail_write)
        with pytest.raises(OSError, match="synthetic point write failure"):
            _run(output, np.arange(7))


def test_model_book_has_analogous_point_selectors_and_independent_selections():
    book = foxes.ModelBook()
    for suffix in ("", "_default", "_all", "_amb", "_amb_all", "_waked_default"):
        output = book.outputs[f"point_results{suffix}"]
        assert isinstance(output, PointResultsOutput)
    for suffix, cls in [
        ("_write_n2", PointResultsCountWriter),
        ("_write_chunks", PointResultsChunkWriter),
        *[
            (f"_write_{period}", PointResultsPeriodWriter)
            for period in ("day", "week", "month", "year")
        ],
    ]:
        assert isinstance(book.outputs[f"point_results{suffix}"], cls)
    assert book.outputs["point_results_all"].variables is None
    default = book.outputs["point_results"]
    assert set(default.variables) == {
        FV.WD,
        FV.WS,
        FV.TI,
        FV.RHO,
        FV.AMB_WD,
        FV.AMB_WS,
        FV.AMB_TI,
        FV.AMB_RHO,
        FV.WEIGHT,
    }
    default.variables.append("custom")
    assert "custom" not in book.outputs["point_results_default"].variables
    assert "custom" not in book.outputs["point_results_write_chunks"].variables
    assert "custom" not in foxes.ModelBook().outputs["point_results"].variables


def _algorithm():
    """Create a small real wake calculation with timestamped synthetic inflow."""
    table = pd.DataFrame(
        {FV.WS: [7.0, 8.0, 9.0, 10.0, 11.0], FV.WD: [90.0, 270.0, 90.0, 270.0, 90.0]},
        index=pd.date_range("2021-01-01", periods=5, freq="12h"),
    )
    states = foxes.input.states.StatesTable(
        table,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
    )
    farm = foxes.WindFarm()
    farm.add_turbine(foxes.Turbine([0.0, 0.0], turbine_models=["NREL5MW"]), verbosity=0)
    return foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model="centre",
        wake_models=["Jensen_linear_k007"],
        verbosity=0,
    )


@pytest.mark.parametrize("backend", ["numpy", "threads", "process"])
@pytest.mark.parametrize("state_size", [1, 3])
@pytest.mark.parametrize(
    "writer_type",
    [PointResultsChunkWriter, PointResultsCountWriter, PointResultsPeriodWriter],
)
def test_real_point_writers_follow_engine_chunks(
    tmp_path, backend, state_size, writer_type
):
    algo = _algorithm()
    options = dict(
        base_name="points",
        variables=[FV.WS],
        out_dir=tmp_path,
        write_nc_pars={"verbosity": 0},
    )
    if writer_type is PointResultsCountWriter:
        options["n_states_per_file"] = 2
    elif writer_type is PointResultsPeriodWriter:
        options["period"] = "day"
    points = np.array([[400.0, 0.0, 90.0], [600.0, 20.0, 90.0], [800.0, 0.0, 90.0]])
    with foxes.Engine.new(
        backend,
        n_procs=2,
        chunk_size_states=state_size,
        chunk_size_points=2,
        verbosity=0,
    ):
        farm_results = algo.calc_farm(outputs="farm_results_all")
        result = algo.calc_points(
            farm_results, points, outputs=["point_results", writer_type(**options)]
        )
    reference = result[FC.POINT_RESULTS][FV.WS]
    datasets = _read_files(tmp_path)
    if writer_type is PointResultsChunkWriter:
        slabs = []
        for state in range((5 + state_size - 1) // state_size):
            tiles = [ds for ds in datasets if ds.attrs["chunki_states"] == state]
            slabs.append(xr.concat(tiles, FC.TARGET))
        combined = xr.concat(slabs, FC.STATE)
    else:
        assert [ds.sizes[FC.STATE] for ds in datasets] == [2, 2, 1]
        combined = xr.concat(datasets, FC.STATE)
    np.testing.assert_allclose(combined[FV.WS].values[..., 0], reference.values)
    np.testing.assert_array_equal(combined[FC.STATE], reference[FC.STATE])
    assert reference.dims == (FC.STATE, FC.POINT)


@pytest.mark.parametrize(
    "selector", ["point_results", "point_results_all", "point_results_write_n2"]
)
def test_precomputed_points_use_named_variable_sets_and_skip_science(
    tmp_path, selector
):
    algo = _algorithm()
    labels = pd.date_range("2021-01-01", periods=5, freq="12h")
    shape = (5, 3)
    fields = {
        v: ((FC.STATE, FC.POINT), np.full(shape, i + 1.0))
        for i, v in enumerate(
            [
                FV.WS,
                FV.WD,
                FV.TI,
                FV.RHO,
                FV.AMB_WS,
                FV.AMB_WD,
                FV.AMB_TI,
                FV.AMB_RHO,
                FV.WEIGHT,
                "custom",
            ]
        )
    }
    data = xr.Dataset(
        fields,
        coords={
            FC.STATE: labels,
            FC.POINT: [0, 1, 2],
            FV.X: (FC.POINT, [1.0, 2.0, 3.0]),
            FV.Y: (FC.POINT, [0.0] * 3),
            FV.H: (FC.POINT, [90.0] * 3),
        },
    )
    output = algo.mbook.outputs[selector]
    output.out_dir = tmp_path
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=2, verbosity=0
    ):
        result = algo.calc_outputs(outputs=selector, point_results=data)
    if selector == "point_results_write_n2":
        assert result == {}
        assert len(_read_files(tmp_path, "point_results")) == 3
    else:
        assert ("custom" in result) is (selector == "point_results_all")
        np.testing.assert_array_equal(result[FV.WS], data[FV.WS])
