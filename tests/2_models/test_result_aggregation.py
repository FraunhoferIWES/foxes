"""Physical result aggregation, compact worker statistics and domain ownership."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.core import FData, MData, OutputList
from foxes.models.outputs import (
    FarmResultsOutput,
    FarmResultsCountWriter,
    FarmResultsChunkWriter,
    FarmResultsPeriodWriter,
    PointResultsOutput,
    PointResultsCountWriter,
    PointResultsChunkWriter,
    PointResultsPeriodWriter,
)


_POWER = np.array(
    [[10.0, 100.0, 30.0], [20.0, 10.0, 20.0], [30.0, 100.0, 10.0], [40.0, 10.0, 0.0]]
)
_WEIGHTS = np.array(
    [[0.1, 0.4, 0.25], [0.2, 0.1, 0.25], [0.3, 0.4, 0.25], [0.4, 0.1, 0.25]]
)
_VARIABLES = [FV.P, FV.AMB_P, FV.CAP, FV.EFF, FV.CAPF, FV.YLD, FV.WEIGHT]


def _farm_data(weights=_WEIGHTS, power=_POWER):
    """Provide immutable, heterogeneous turbine arrays and original memberships."""
    data = {
        FV.P: power,
        FV.AMB_P: power + 20,
        FV.CAP: np.broadcast_to([100.0, 200.0, 300.0], power.shape),
        FV.EFF: np.full_like(power, 999.0),
        FV.CAPF: np.full_like(power, 999.0),
        FV.YLD: power * 0.00876,
        FV.WEIGHT: weights,
    }
    farm = SimpleNamespace(
        turbine_names=["T0", "T1", "T2"],
        wind_farm_list=["north", "north", None],
        cluster_list=["a", "b", "a"],
    )
    algo = SimpleNamespace(
        farm=farm,
        verbosity=0,
        farm_controller=SimpleNamespace(
            turbine_types=[SimpleNamespace(P_unit="kW") for _ in range(3)]
        ),
    )
    return algo, data


def _farm_run(output, sizes=(1, 2, 1), indices=None, weights=_WEIGHTS, power=_POWER):
    """Run prepared farm chunks without relying on scientific models."""
    algo, data = _farm_data(weights, power)
    indices = np.arange(4) if indices is None else np.asarray(indices)
    results = foxes.utils.Dict()
    start = 0
    for index, size in enumerate(sizes):
        selected = indices[start : start + size]
        mdata = MData(
            {
                FC.STATE: np.asarray(
                    ["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"],
                    dtype="datetime64[ns]",
                )[selected]
            },
            dims={FC.STATE: (FC.STATE,)},
            states_i0=start,
            chunki_states=index,
            chunki_points=0,
            n_chunks_states=len(sizes),
            n_chunks_points=1,
        )
        values = {FC.STATE: mdata[FC.STATE], FC.TURBINE: np.arange(3)}
        values.update({name: array[selected].copy() for name, array in data.items()})
        fdata = FData(
            values,
            dims={
                FC.STATE: (FC.STATE,),
                FC.TURBINE: (FC.TURBINE,),
                **{name: (FC.STATE, FC.TURBINE) for name in data},
            },
        )
        for array in fdata.values():
            array.flags.writeable = False
        calculated = output.calculate(algo, mdata, fdata, None, {})
        pending = foxes.utils.Dict(
            {key: {(index, 0): payload} for key, payload in calculated.items()}
        )
        output.update_results(algo, len(sizes), 1, pending, results)
        start += size
    output.finalize_results(algo, len(sizes), 1, results)
    return results, algo


@pytest.mark.parametrize("aggregate", [None, "wind_farm", "cluster", "all"])
@pytest.mark.parametrize("states", [False, True])
@pytest.mark.parametrize("sizes", [(4,), (1, 2, 1)])
def test_farm_physical_reductions_and_independent_weights(
    tmp_path, aggregate, states, sizes
):
    output = FarmResultsOutput(
        variables=_VARIABLES,
        aggregate_turbines=aggregate,
        aggregate_states=states,
        out_dir=tmp_path,
    )
    results, _ = _farm_run(output, sizes)
    result = next(iter(results.values()))
    groups = {
        "wind_farm": [[0, 1], [2]],
        "cluster": [[0, 2], [1]],
        "all": [[0, 1, 2]],
        None: [[0], [1], [2]],
    }[aggregate]
    power = np.sum(_POWER * _WEIGHTS, axis=0) if states else _POWER
    ambient = np.sum((_POWER + 20) * _WEIGHTS, axis=0) if states else _POWER + 20
    expected_p = np.stack([power[..., group].sum(axis=-1) for group in groups], axis=-1)
    expected_a = np.stack(
        [ambient[..., group].sum(axis=-1) for group in groups], axis=-1
    )
    cap = np.asarray([sum(np.array([100.0, 200.0, 300.0])[group]) for group in groups])
    if aggregate == "all":
        expected_p, expected_a, cap = expected_p[..., 0], expected_a[..., 0], cap[0]
    np.testing.assert_allclose(result[FV.P], expected_p)
    if aggregate is not None or states:
        np.testing.assert_allclose(result[FV.EFF], expected_p / expected_a)
        np.testing.assert_allclose(result[FV.CAPF], expected_p / cap)
    np.testing.assert_allclose(result[FV.YLD], expected_p * 0.00876)
    assert (FC.STATE in result.dims) is (not states)
    assert (FC.TURBINE in result.dims) is (aggregate is None)
    if aggregate == "wind_farm":
        np.testing.assert_array_equal(result[FC.FARM], ["north", "<unassigned>"])
    if aggregate == "cluster":
        np.testing.assert_array_equal(result[FC.CLUSTER], ["a", "b"])


def test_partial_state_selection_normalizes_each_turbine(tmp_path):
    output = FarmResultsOutput(
        variables=[FV.P, FV.EFF, FV.CAPF],
        aggregate_turbines="all",
        aggregate_states=True,
        out_dir=tmp_path,
    )
    result = next(iter(_farm_run(output, (1, 2), indices=[1, 2, 3])[0].values()))
    mean = np.sum(_POWER[1:] * _WEIGHTS[1:], axis=0) / np.sum(_WEIGHTS[1:], axis=0)
    ambient = np.sum((_POWER[1:] + 20) * _WEIGHTS[1:], axis=0) / np.sum(
        _WEIGHTS[1:], axis=0
    )
    assert float(result[FV.P]) == pytest.approx(mean.sum())
    assert float(result[FV.EFF]) == pytest.approx(mean.sum() / ambient.sum())
    assert float(result[FV.CAPF]) == pytest.approx(mean.sum() / 600)
    assert set(result.data_vars) == {FV.P, FV.EFF, FV.CAPF}


def test_state_workers_eliminate_history_before_transport(tmp_path):
    output = FarmResultsOutput(
        variables=[FV.P, FV.EFF],
        aggregate_turbines="all",
        aggregate_states=True,
        out_dir=tmp_path,
    )
    algo, data = _farm_data()
    mdata = MData(
        {FC.STATE: np.arange(4)}, dims={FC.STATE: (FC.STATE,)}, chunki_points=0
    )
    fdata = FData(
        {FC.STATE: np.arange(4), FC.TURBINE: np.arange(3), **data},
        dims={
            FC.STATE: (FC.STATE,),
            FC.TURBINE: (FC.TURBINE,),
            **{v: (FC.STATE, FC.TURBINE) for v in data},
        },
    )
    payload = next(iter(output.calculate(algo, mdata, fdata, None, {}).values()))
    assert FC.STATE not in payload["coords"]
    assert all(
        FC.STATE not in dims and array.shape == (3,)
        for dims, array in payload["data_vars"].values()
    )


@pytest.mark.parametrize(
    "weights", [np.full((4, 3), -1.0), np.full((4, 3), np.nan), np.zeros((4, 3))]
)
def test_state_aggregation_rejects_invalid_weights(tmp_path, weights):
    output = FarmResultsOutput(
        variables=[FV.P], aggregate_states=True, out_dir=tmp_path
    )
    with pytest.raises(ValueError, match="weight"):
        _farm_run(output, weights=weights)


def test_zero_weight_nan_power_is_ignored(tmp_path):
    power, weights = _POWER.copy(), _WEIGHTS.copy()
    power[0, 0], weights[0, 0] = np.nan, 0
    output = FarmResultsOutput(
        variables=[FV.P], aggregate_states=True, out_dir=tmp_path
    )
    with np.errstate(all="raise"):
        result = next(iter(_farm_run(output, power=power, weights=weights)[0].values()))
    assert float(result[FV.P][0]) == pytest.approx(
        np.dot(_POWER[1:, 0], weights[1:, 0]) / weights[:, 0].sum()
    )


def test_aggregation_outputs_compose_without_consuming_each_other(tmp_path):
    chain = OutputList(
        outputs=[
            FarmResultsOutput(variables=[FV.P], out_dir=tmp_path),
            FarmResultsOutput(
                variables=[FV.P], aggregate_turbines="wind_farm", out_dir=tmp_path
            ),
            FarmResultsOutput(
                variables=[FV.EFF],
                aggregate_turbines="all",
                aggregate_states=True,
                out_dir=tmp_path,
            ),
        ]
    )
    results, _ = _farm_run(chain)
    assert set(results) == {
        FC.FARM_RESULTS,
        "farm_results_aggregate_wind_farm",
        "farm_results_aggregate_all_mean",
    }
    assert results[FC.FARM_RESULTS].sizes[FC.TURBINE] == 3


@pytest.mark.parametrize(
    "writer", [FarmResultsCountWriter, FarmResultsChunkWriter, FarmResultsPeriodWriter]
)
@pytest.mark.parametrize("states", [False, True])
@pytest.mark.parametrize("sizes", [(4,), (1, 2, 1), (2, 2)])
def test_farm_writers_reduce_each_file_interval(tmp_path, writer, states, sizes):
    options = dict(
        base_name="farm",
        variables=[FV.P, FV.EFF],
        aggregate_turbines="all",
        aggregate_states=states,
        out_dir=tmp_path,
        write_nc_pars={"verbosity": 0},
    )
    if writer is FarmResultsCountWriter:
        options["n_states_per_file"] = 2
    elif writer is FarmResultsPeriodWriter:
        options["period"] = "day"
    with foxes.Engine.new("numpy", verbosity=0):
        result, _ = _farm_run(writer(**options), sizes=sizes)
    assert result == {}
    files = sorted(tmp_path.glob("*.nc"))
    assert len(files) == (len(sizes) if writer is FarmResultsChunkWriter else 2)
    edges = (
        np.r_[0, np.cumsum(sizes)] if writer is FarmResultsChunkWriter else [0, 2, 4]
    )
    for index, path in enumerate(files):
        with xr.open_dataset(path) as data:
            p = _POWER[edges[index] : edges[index + 1]]
            w = _WEIGHTS[edges[index] : edges[index + 1]]
            expected = (p * w).sum(axis=0) / w.sum(axis=0) if states else p
            np.testing.assert_allclose(data[FV.P], expected.sum(axis=-1))
            assert (FC.STATE in data.dims) is (not states)


def _algorithm(algorithm="Downwind", index=None):
    states = foxes.input.states.StatesTable(
        pd.DataFrame({FV.WS: [7.0, 8.0, 9.0, 10.0], FV.WD: [270.0] * 4}, index=index),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
    )
    farm = foxes.WindFarm()
    for index in range(3):
        farm.add_turbine(
            foxes.Turbine(
                [index * 500.0, 0.0],
                turbine_models=["NREL5MW"],
                wind_farm_name="north" if index < 2 else None,
            ),
            verbosity=0,
        )
    return getattr(foxes.algorithms, algorithm)(
        farm,
        states,
        wake_models=["Jensen_linear_k007"],
        rotor_model="centre",
        verbosity=0,
    )


@pytest.mark.parametrize("backend", ["numpy", "threads", "process"])
def test_collected_group_memberships_support_output_and_point_reuse(backend):
    algo = _algorithm()
    for turbine in algo.farm.turbines:
        turbine.cluster_name = "coastal"
    with foxes.Engine.new(backend, n_procs=2, chunk_size_states=2, verbosity=0):
        raw = algo.calc_farm(outputs="farm_results_all")
        original = raw.copy(deep=True)
        summary = algo.calc_outputs(
            farm_results=raw,
            outputs=FarmResultsOutput(
                variables=[FV.P], aggregate_turbines="wind_farm", aggregate_states=True
            ),
        )
        points = algo.calc_points(raw, points=np.array([[1500.0, 0.0, 90.0]]))

    assert raw[FC.FARM].dims == (FC.TURBINE,)
    assert raw[FC.CLUSTER].dims == (FC.TURBINE,)
    expected = raw[FV.P].mean(FC.STATE).values
    np.testing.assert_allclose(summary[FV.P], [expected[:2].sum(), expected[2]])
    assert points[FV.WS].shape == (4, 1)
    assert np.isfinite(points[FV.WS]).all()
    xr.testing.assert_identical(raw, original)


@pytest.mark.parametrize("backend", ["numpy", "threads", "process"])
@pytest.mark.parametrize("algorithm", ["Downwind", "Iterative"])
def test_real_aggregated_metrics_use_selected_output_evaluation(
    tmp_path, backend, algorithm
):
    variables = [FV.P, FV.AMB_P, FV.CAP, FV.EFF, FV.CAPF, FV.YLD, FV.WEIGHT]
    with foxes.Engine.new(backend, n_procs=2, chunk_size_states=2, verbosity=0):
        raw = _algorithm(algorithm).calc_farm(
            outputs=FarmResultsOutput(variables=variables)
        )
        result = _algorithm(algorithm).calc_farm(
            outputs=FarmResultsOutput(
                variables=variables, aggregate_turbines="all", aggregate_states=True
            )
        )
    weights = raw[FV.WEIGHT].broadcast_like(raw[FV.P]).values
    mean = (raw[FV.P].values * weights).sum(axis=0) / weights.sum(axis=0)
    ambient = (raw[FV.AMB_P].values * weights).sum(axis=0) / weights.sum(axis=0)
    assert float(result[FV.P]) == pytest.approx(mean.sum())
    assert float(result[FV.EFF]) == pytest.approx(mean.sum() / ambient.sum())
    assert float(result[FV.YLD]) == pytest.approx(mean.sum() * 0.00876)


def _points():
    values = np.arange(12.0).reshape(4, 3) + 1
    return xr.Dataset(
        {
            FV.WS: ((FC.STATE, FC.POINT), values),
            FV.WEIGHT: ((FC.STATE, FC.POINT), _WEIGHTS),
        },
        coords={
            FC.STATE: np.arange(4),
            FC.POINT: [0, 1, 2],
            FV.X: (FC.POINT, [0.0, 10.0, 20.0]),
            FV.Y: (FC.POINT, [0.0] * 3),
            FV.H: (FC.POINT, [90.0] * 3),
        },
    )


@pytest.mark.parametrize("backend", ["numpy", "threads", "process"])
@pytest.mark.parametrize("aggregate", [None, "all"])
@pytest.mark.parametrize("states", [False, True])
def test_point_aggregation_handles_uneven_tiles_and_per_point_weights(
    tmp_path, backend, aggregate, states
):
    algo, data = _algorithm(), _points()
    output = PointResultsOutput(
        variables=[FV.WS],
        aggregate_points=aggregate,
        aggregate_states=states,
        out_dir=tmp_path,
    )
    with foxes.Engine.new(
        backend, n_procs=2, chunk_size_states=3, chunk_size_points=2, verbosity=0
    ):
        result = algo.calc_outputs(outputs=output, point_results=data)
    expected = (
        (data[FV.WS].values * _WEIGHTS).sum(axis=0) if states else data[FV.WS].values
    )
    if aggregate:
        expected = expected.mean(axis=-1)
    np.testing.assert_allclose(result[FV.WS], expected)
    assert (FC.STATE in result.dims) is (not states)
    assert (FC.POINT in result.dims) is (aggregate is None)


@pytest.mark.parametrize(
    "writer",
    [PointResultsCountWriter, PointResultsChunkWriter, PointResultsPeriodWriter],
)
@pytest.mark.parametrize("states", [False, True])
def test_point_writers_accept_all_target_and_state_aggregation(
    tmp_path, writer, states
):
    data = _points()
    data = data.assign_coords(
        {FC.STATE: pd.date_range("2021-01-01", periods=4, freq="12h")}
    )
    algo = _algorithm(index=data[FC.STATE].values)
    options = dict(
        base_name="point",
        variables=[FV.WS],
        aggregate_points="all",
        aggregate_states=states,
        out_dir=tmp_path,
        write_nc_pars={"verbosity": 0},
    )
    if writer is PointResultsCountWriter:
        options["n_states_per_file"] = 2
    elif writer is PointResultsPeriodWriter:
        options["period"] = "day"
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=2, verbosity=0
    ) as engine:
        _, target_sizes = engine.calc_chunk_sizes(4, 3)
        assert algo.calc_outputs(outputs=writer(**options), point_results=data) == {}
    files = sorted(tmp_path.glob("*.nc"))
    assert len(files) == (4 if writer is PointResultsChunkWriter else 2)
    for index, path in enumerate(files):
        with xr.open_dataset(path) as result:
            assert FC.TARGET not in result.dims
            assert (FC.STATE in result.dims) is (not states)
            slab = index // 2 if writer is PointResultsChunkWriter else index
            edges = np.r_[0, np.cumsum(target_sizes)]
            targets = (
                slice(edges[index % 2], edges[index % 2 + 1])
                if writer is PointResultsChunkWriter
                else slice(None)
            )
            p = data[FV.WS].values[slab * 2 : slab * 2 + 2, targets]
            w = _WEIGHTS[slab * 2 : slab * 2 + 2, targets]
            expected = (p * w).sum(axis=0) / w.sum(axis=0) if states else p
            np.testing.assert_allclose(
                result[FV.WS].squeeze(FC.TPOINT), expected.mean(axis=-1)
            )


@pytest.mark.parametrize("states", [False, True])
@pytest.mark.parametrize("variables", [[FV.EFF], [FV.CAPF], None])
def test_ratio_only_and_unrestricted_selections_retain_ratios(
    tmp_path, states, variables
):
    output = FarmResultsOutput(
        variables=variables,
        aggregate_turbines="all",
        aggregate_states=states,
        out_dir=tmp_path,
    )
    result = next(iter(_farm_run(output)[0].values()))
    if variables is not None:
        assert set(result.data_vars) == set(variables)
    p = (_POWER * _WEIGHTS).sum(axis=0) if states else _POWER
    a = ((_POWER + 20) * _WEIGHTS).sum(axis=0) if states else _POWER + 20
    if variables is None or FV.EFF in variables:
        np.testing.assert_allclose(result[FV.EFF], p.sum(axis=-1) / a.sum(axis=-1))
    if variables is None or FV.CAPF in variables:
        np.testing.assert_allclose(result[FV.CAPF], p.sum(axis=-1) / 600)


def test_filtered_ambient_power_remains_an_efficiency_dependency(tmp_path):
    output = FarmResultsOutput(
        variables=[FV.EFF, FV.AMB_P],
        ambient_vars=False,
        aggregate_turbines="all",
        out_dir=tmp_path,
    )
    result = next(iter(_farm_run(output)[0].values()))
    assert set(result.data_vars) == {FV.EFF}
    np.testing.assert_allclose(
        result[FV.EFF], _POWER.sum(axis=-1) / (_POWER + 20).sum(axis=-1)
    )


@pytest.mark.parametrize("writer", [FarmResultsCountWriter, FarmResultsPeriodWriter])
def test_interval_writer_workers_return_only_per_file_statistics(tmp_path, writer):
    options = dict(
        base_name="farm", variables=[FV.P], aggregate_states=True, out_dir=tmp_path
    )
    options.update(
        n_states_per_file=2
    ) if writer is FarmResultsCountWriter else options.update(period="day")
    output = writer(**options)
    algo, fields = _farm_data()
    states = np.array(
        ["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"], dtype="datetime64[ns]"
    )
    mdata = MData(
        {FC.STATE: states}, dims={FC.STATE: (FC.STATE,)}, states_i0=0, chunki_points=0
    )
    fdata = FData(
        {FC.STATE: states, FC.TURBINE: np.arange(3), **fields},
        dims={
            FC.STATE: (FC.STATE,),
            FC.TURBINE: (FC.TURBINE,),
            **{v: (FC.STATE, FC.TURBINE) for v in fields},
        },
    )
    payload = next(iter(output.calculate(algo, mdata, fdata, None, {}).values()))
    assert len(payload["intervals"]) == 2
    for interval in payload["intervals"]:
        assert FC.STATE not in interval["coords"]
        assert all(
            FC.STATE not in dims and array.shape == (3,)
            for dims, array in interval["data_vars"].values()
        )


@pytest.mark.parametrize("kind", ["wind_farm", "cluster", "bad"])
def test_points_reject_undefined_memberships(kind):
    with pytest.raises(ValueError, match="point aggregation"):
        PointResultsOutput(aggregate_points=kind)


def test_native_mixed_units_are_rejected_within_a_group(tmp_path):
    algo, data = _farm_data()
    algo.farm_controller.turbine_types[1].P_unit = "MW"
    output = FarmResultsOutput(
        variables=[FV.P], aggregate_turbines="all", out_dir=tmp_path
    )
    mdata = MData(
        {FC.STATE: np.arange(4)}, dims={FC.STATE: (FC.STATE,)}, chunki_points=0
    )
    fdata = FData(
        {FC.STATE: np.arange(4), FC.TURBINE: np.arange(3), **data},
        dims={
            FC.STATE: (FC.STATE,),
            FC.TURBINE: (FC.TURBINE,),
            **{v: (FC.STATE, FC.TURBINE) for v in data},
        },
    )
    with pytest.raises(ValueError, match="mixed turbine power units"):
        output.calculate(algo, mdata, fdata, None, {})


def test_registry_replaces_generic_writer_with_collection_aggregation():
    book = foxes.ModelBook()
    assert "results_writer" not in book.outputs
    assert not hasattr(foxes.models.outputs, "ResultsWriter")
    for name in (
        "farms_results",
        "cluster_results_mean",
        "total_results_mean",
        "farm_results_mean",
        "point_results_aggregate_all",
        "point_results_mean",
    ):
        assert isinstance(book.outputs[name], (FarmResultsOutput, PointResultsOutput))


@pytest.mark.parametrize(
    "collector,parameter",
    [
        (FarmResultsOutput, "aggregate_turbines"),
        (PointResultsOutput, "aggregate_points"),
    ],
)
def test_domain_parameter_rename_has_no_aggregate_alias(collector, parameter):
    output = collector(**{parameter: "all"})
    assert getattr(output, parameter) == "all"
    assert not hasattr(output, "aggregate")
    with pytest.raises(TypeError, match="aggregate"):
        collector(aggregate="all")


def test_precomputed_direction_means_normalize_tiny_selected_masses(tmp_path):
    data = _points().assign(
        {
            FV.WD: ((FC.STATE, FC.POINT), np.full((4, 3), 350.0)),
            FV.WEIGHT: ((FC.STATE, FC.POINT), _WEIGHTS * 1e-15),
        }
    )
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=2, verbosity=0
    ):
        result = _algorithm().calc_outputs(
            point_results=data,
            outputs=PointResultsOutput(
                variables=[FV.WD],
                aggregate_points="all",
                aggregate_states=True,
                out_dir=tmp_path,
            ),
        )
    np.testing.assert_allclose(result[FV.WD], 350.0)


def test_circular_direction_means_and_cancellation(tmp_path):
    directions = np.array(
        [
            [350.0, 10.0, 0.0],
            [0.0, 120.0, 240.0],
            [10.0, 350.0, 0.0],
            [90.0, 90.0, 90.0],
        ]
    )
    data = _points().assign({FV.WD: ((FC.STATE, FC.POINT), directions)})
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=2, verbosity=0
    ):
        result = _algorithm().calc_outputs(
            point_results=data,
            outputs=PointResultsOutput(
                variables=[FV.WD],
                aggregate_points="all",
                out_dir=tmp_path,
            ),
        )
    np.testing.assert_allclose(
        result[FV.WD], [0.0, np.nan, 0.0, 90.0], atol=1e-12, equal_nan=True
    )


def test_aggregated_efficiency_preserves_zero_division_contract(tmp_path):
    algo, _ = _farm_data()
    power = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [np.nan, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    output = FarmResultsOutput(
        variables=[FV.EFF], aggregate_turbines="all", out_dir=tmp_path
    )
    with np.errstate(all="raise"):
        payload = output._file_payload(
            algo,
            dict(
                coords={FC.STATE: np.arange(4), FC.TURBINE: np.arange(3)},
                data_vars={
                    FV.P: ((FC.STATE, FC.TURBINE), power),
                    FV.AMB_P: ((FC.STATE, FC.TURBINE), np.zeros_like(power)),
                },
                attrs={},
            ),
        )
    np.testing.assert_allclose(
        payload["data_vars"][FV.EFF][1], [0.0, np.inf, np.nan, np.inf], equal_nan=True
    )


@pytest.mark.parametrize(
    "prefix,parameter",
    [
        ("farm_results", "aggregate_turbines"),
        ("point_results", "aggregate_points"),
    ],
)
def test_preset_mutation_updates_state_weight_requirements(prefix, parameter):
    output = foxes.ModelBook().outputs[prefix]
    setattr(output, parameter, "all")
    output.aggregate_states = True
    expected = (
        "farm_results_aggregate_all_mean"
        if prefix == "farm_results"
        else "point_results_aggregate_all_mean"
    )
    assert output._public_key == expected
    assert FV.WEIGHT in (
        output.fdata_variables if prefix == "farm_results" else output.tdata_variables
    )


@pytest.mark.parametrize("name", [None, "custom_points", FC.POINT_RESULTS])
@pytest.mark.parametrize("aggregate", [None, "all"])
@pytest.mark.parametrize("states", [False, True])
def test_point_public_dimensions_do_not_depend_on_result_name(name, aggregate, states):
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=2, verbosity=0
    ):
        result = _algorithm().calc_outputs(
            point_results=_points(),
            outputs=PointResultsOutput(
                variables=[FV.WS],
                aggregate_points=aggregate,
                aggregate_states=states,
                result_name=name,
            ),
        )
    assert FC.TARGET not in result.dims
    assert FC.TPOINT not in result.dims
    assert (FC.POINT in result.dims) is (aggregate is None)
    assert (FC.STATE in result.dims) is (not states)


def test_farm_selected_field_list_stays_editable_before_grouping():
    output = FarmResultsOutput(variables=[FV.P])
    output.fdata_variables.append(FV.EFF)
    with foxes.Engine.new("numpy", verbosity=0):
        raw = _algorithm().calc_farm(outputs=output)
        assert FV.EFF in raw
    output.aggregate_turbines = "all"
    output.aggregate_states = True
    with foxes.Engine.new("numpy", verbosity=0):
        result = _algorithm().calc_farm(outputs=output)
    assert set(result.data_vars) == {FV.P, FV.EFF}
    assert not result.dims


def test_point_variable_edits_update_reduction_requirements():
    output = PointResultsOutput(variables=[FV.WS])
    output.tdata_variables.append(FV.WD)
    output.aggregate_points = "all"
    output.aggregate_states = True
    assert output.tdata_variables == [FV.WS, FV.WD, FV.WEIGHT]
