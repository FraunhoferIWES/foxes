"""Worker payloads retain only data required by each output's final contract."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.models.outputs import (
    FarmLayoutOutput,
    FlowPlots2D,
    PointResultsOutput,
    RosePlotOutput,
    WindRoseBinPlot,
)


@pytest.fixture
def farm_data():
    count = 7
    states = pd.date_range("2026-01-01", periods=count, freq="h")
    values = np.arange(count * 2, dtype=float).reshape(count, 2) + 1
    weights = np.array([0.05, 0.1, 0.15, 0.2, 0.05, 0.25, 0.2])
    return xr.Dataset(
        {
            FV.P: ((FC.STATE, FC.TURBINE), values),
            FV.AMB_P: ((FC.STATE, FC.TURBINE), values + 3),
            FV.REWS: ((FC.STATE, FC.TURBINE), values / 3),
            FV.AMB_REWS: ((FC.STATE, FC.TURBINE), values / 3 + 1),
            FV.AMB_WD: ((FC.STATE, FC.TURBINE), np.tile([355.0, 10.0], (count, 1))),
            FV.WEIGHT: ((FC.STATE,), weights),
            FV.CAP: ((FC.TURBINE,), [50.0, 100.0]),
            FV.X: ((FC.STATE, FC.TURBINE), values * 100),
            FV.Y: ((FC.STATE, FC.TURBINE), values * 10),
            FV.H: ((FC.STATE, FC.TURBINE), values + 90),
        },
        coords={FC.STATE: states, FC.TURBINE: [0, 1]},
    )


@pytest.fixture
def algorithm(farm_data):
    states = foxes.input.states.StatesTable(
        pd.DataFrame(
            {FV.WS: np.ones(farm_data.sizes[FC.STATE])},
            index=farm_data[FC.STATE].values,
        ),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.WD: 270.0, FV.TI: 0.08, FV.RHO: 1.225},
    )
    farm = foxes.WindFarm()
    for i in range(2):
        farm.add_turbine(
            foxes.Turbine(
                [i * 600.0, 0.0], H=90.0, D=100.0, turbine_models=["NREL5MW"]
            ),
            verbosity=0,
        )
    algo = foxes.algorithms.Downwind(farm, states, wake_models=[], verbosity=0)
    algo.initialize()
    return algo


@pytest.mark.parametrize(
    "contraction", ["frequency", "weights", "mean_no_weights", "sum_no_weights"]
)
@pytest.mark.parametrize("chunk_size", [1, 3, 7])
def test_rose_payloads_contain_bins_not_state_samples(
    algorithm, farm_data, monkeypatch, contraction, chunk_size
):
    parameters = dict(ws_bins=[0, 2, 4, 6, 8], wd_sectors=8, plot=False)
    if contraction == "frequency":
        output = RosePlotOutput(**parameters)
        north = [5.0, 45.0, 50.0, 0.0]
    else:
        output = WindRoseBinPlot(variable=FV.P, contraction=contraction, **parameters)
        north = {
            "weights": [1.0, 2.45 / 0.45, 11.6, np.nan],
            "mean_no_weights": [1.0, 5.0, 11.0, np.nan],
            "sum_no_weights": [1.0, 15.0, 33.0, 0.0],
        }[contraction]
    empty = np.nan if contraction in {"weights", "mean_no_weights"} else 0.0
    expected = np.full((8, 4), empty)
    expected[0] = north
    calculate = type(output).calculate
    inspected = []

    def capture(self, *args, **kwargs):
        result = calculate(self, *args, **kwargs)
        payload = result[self.var("data")]
        assert set(payload) <= {"sum", "weight"}
        assert all(values.shape == (8, 4) for values in payload.values())
        assert sum(values.nbytes for values in payload.values()) <= 2 * 8 * 4 * 8
        inspected.append(payload)
        return result

    monkeypatch.setattr(type(output), "calculate", capture)
    with foxes.Engine.new("numpy", chunk_size_states=chunk_size, verbosity=0):
        result = algorithm.calc_outputs(outputs=output, farm_results=farm_data)
    assert inspected
    variable = "frequency" if contraction == "frequency" else FV.P
    np.testing.assert_allclose(result[variable], expected)
    assert result[variable].dims == (FV.AMB_WD, FV.AMB_REWS)
    np.testing.assert_array_equal(result[FV.AMB_WD], np.arange(0, 360, 45))
    np.testing.assert_array_equal(result[FV.AMB_REWS], [1, 3, 5, 7])
    np.testing.assert_array_equal(
        result.attrs[f"{FV.AMB_WD}_bounds"], np.arange(-22.5, 360, 45)
    )
    np.testing.assert_array_equal(
        result.attrs[f"{FV.AMB_REWS}_bounds"], [0, 2, 4, 6, 8]
    )


@pytest.mark.parametrize("position", [0, 4, -1, -5])
def test_layout_payload_returns_only_selected_geometry(
    algorithm, farm_data, monkeypatch, position
):
    output = FarmLayoutOutput(
        operation="get_layout_data", from_results=True, results_state=position
    )
    calculate = FarmLayoutOutput.calculate
    inspected = []

    def capture(self, *args, **kwargs):
        result = calculate(self, *args, **kwargs)
        payload = result[self.var("data")]
        assert "data_vars" not in payload
        for geometry in payload["geometry"].values():
            assert set(geometry) == {FV.X, FV.Y, FV.H}
            assert all(values.shape == (2,) for values in geometry.values())
        inspected.append(payload)
        return result

    monkeypatch.setattr(FarmLayoutOutput, "calculate", capture)
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        result = algorithm.calc_outputs(outputs=output, farm_results=farm_data)
    expected = np.stack(
        [farm_data[v].values[position] for v in (FV.X, FV.Y, FV.H)], axis=-1
    )
    np.testing.assert_allclose(result, expected)
    assert inspected


def test_algorithm_layout_output_sends_no_scientific_arrays(algorithm, monkeypatch):
    output = FarmLayoutOutput(operation="get_layout_data")
    calculate = FarmLayoutOutput.calculate
    inspected = []

    def capture(self, *args, **kwargs):
        result = calculate(self, *args, **kwargs)
        assert self.var("data") not in result
        inspected.append(True)
        return result

    monkeypatch.setattr(FarmLayoutOutput, "calculate", capture)
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        result = algorithm.calc_farm(outputs=output)
    assert result.shape == (2, 3)
    assert inspected


@pytest.mark.parametrize("point", [0, 3, 6, -1, -3, -7])
def test_rose_selects_one_precomputed_point_across_tiles(algorithm, farm_data, point):
    count = farm_data.sizes[FC.STATE]
    values = np.arange(count * 7).reshape(count, 7) % 12 + 1.0
    data = xr.Dataset(
        {
            FV.AMB_WS: ((FC.STATE, FC.POINT), values),
            FV.AMB_WD: ((FC.STATE, FC.POINT), values * 27),
            FV.WEIGHT: farm_data[FV.WEIGHT],
        },
        coords={
            FC.STATE: farm_data[FC.STATE],
            FC.POINT: np.arange(7),
            FV.X: (FC.POINT, np.arange(7) * 100.0),
            FV.Y: (FC.POINT, np.zeros(7)),
            FV.H: (FC.POINT, np.full(7, 90.0)),
        },
    )
    output = RosePlotOutput(use_points=True, point=point, ws_var=FV.AMB_WS, plot=False)
    expected = (
        100
        * np.histogram2d(
            values[:, point] * 27,
            values[:, point],
            bins=[np.arange(-15, 360, 30), output.ws_bins],
            weights=farm_data[FV.WEIGHT].values,
        )[0]
    )
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=3, verbosity=0
    ):
        actual = algorithm.calc_outputs(
            outputs=output, farm_results=farm_data.drop_vars(FV.CAP), point_results=data
        )
    np.testing.assert_allclose(actual.frequency, expected)
    assert actual.frequency.dims == (FV.AMB_WD, FV.AMB_WS)


def test_binned_means_apply_small_weight_threshold_after_global_merge(
    algorithm, farm_data
):
    data = farm_data.copy(deep=True)
    data[FV.WEIGHT].values[:] = 6e-14
    output = WindRoseBinPlot(variable=FV.P, ws_bins=[0, 50], wd_sectors=1, plot=False)
    with foxes.Engine.new("numpy", chunk_size_states=1, verbosity=0):
        actual = algorithm.calc_outputs(outputs=output, farm_results=data)
    np.testing.assert_allclose(actual[FV.P], [[7.0]])
    assert np.isfinite(actual[FV.P]).all()


@pytest.mark.parametrize("color", ["mean_P", "sum_P", "min_P", "max_P", FV.CAP])
def test_layout_color_reductions_preserve_global_values(algorithm, farm_data, color):
    import matplotlib.pyplot as plt

    output = FarmLayoutOutput(parameters={"color_by": color})
    values = farm_data[FV.P].values
    if color == FV.CAP:
        expected = farm_data[FV.CAP]
    elif color.startswith("mean_"):
        expected = np.sum(values * farm_data[FV.WEIGHT].values[:, None], axis=0)
    else:
        expected = getattr(np, color.split("_")[0])(values, axis=0)
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        axes = algorithm.calc_outputs(outputs=output, farm_results=farm_data)
    try:
        np.testing.assert_allclose(axes.collections[0].get_array(), expected)
    finally:
        plt.close(axes.figure)


def test_layout_preserves_precomputed_color_field_precedence(algorithm, farm_data):
    import matplotlib.pyplot as plt

    data = farm_data.drop_vars(FV.P)
    data["mean_P"] = ((FC.TURBINE,), [12.0, 34.0])
    output = FarmLayoutOutput(parameters={"color_by": "mean_P"})
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        axes = algorithm.calc_outputs(outputs=output, farm_results=data)
    try:
        np.testing.assert_array_equal(axes.collections[0].get_array(), [12.0, 34.0])
    finally:
        plt.close(axes.figure)


def test_layout_selection_is_relative_to_selected_states(algorithm, farm_data):
    output = FarmLayoutOutput(
        operation="get_layout_data", from_results=True, results_state=-2
    )
    with foxes.Engine.new("numpy", chunk_size_states=1, verbosity=0):
        result = algorithm.calc_outputs(
            outputs=output,
            farm_results=farm_data,
            states_sel=farm_data[FC.STATE].values[[1, 3, 5]],
        )
    np.testing.assert_allclose(result[:, 0], farm_data[FV.X].values[3])


@pytest.mark.parametrize("position", [7, -8])
def test_layout_reports_unavailable_result_state(algorithm, farm_data, position):
    output = FarmLayoutOutput(
        operation="get_layout_data", from_results=True, results_state=position
    )
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        with pytest.raises(IndexError, match="unavailable"):
            algorithm.calc_outputs(outputs=output, farm_results=farm_data)


@pytest.mark.parametrize("moving", [False, True])
def test_point_collector_transports_static_coordinates_once(
    algorithm, farm_data, monkeypatch, moving
):
    states = farm_data[FC.STATE]
    shape = (len(states), 3)
    x = np.broadcast_to(np.arange(3) * 100.0, shape).copy()
    if moving:
        x += np.arange(len(states))[:, None]
    data = xr.Dataset(
        {FV.WS: ((FC.STATE, FC.POINT), np.arange(np.prod(shape)).reshape(shape) + 1.0)},
        coords={
            FC.STATE: states,
            FC.POINT: [0, 1, 2],
            FV.X: ((FC.STATE, FC.POINT), x),
            FV.Y: (FC.POINT, np.zeros(3)),
            FV.H: (FC.POINT, np.full(3, 90.0)),
        },
    )
    output = PointResultsOutput(variables=[FV.WS])
    calculate = PointResultsOutput.calculate
    inspected = []

    def capture(self, *args, **kwargs):
        result = calculate(self, *args, **kwargs)
        payload = result[FC.POINT_RESULTS]
        assert payload["points"].shape[0] == (len(payload["states"]) if moving else 1)
        inspected.append(True)
        return result

    monkeypatch.setattr(PointResultsOutput, "calculate", capture)
    with foxes.Engine.new(
        "numpy", chunk_size_states=3, chunk_size_points=2, verbosity=0
    ):
        result = algorithm.calc_outputs(
            outputs=output, farm_results=farm_data.drop_vars(FV.CAP), point_results=data
        )
    assert inspected
    np.testing.assert_array_equal(result[FV.WS], data[FV.WS])
    np.testing.assert_array_equal(result[FV.X], x)


@pytest.mark.parametrize("backend", ["numpy", "threads", "process"])
def test_composed_compact_outputs_are_backend_independent(
    algorithm, farm_data, backend
):
    outputs = [
        RosePlotOutput(plot=False, result_name="rose"),
        FarmLayoutOutput(
            operation="get_layout_data",
            from_results=True,
            results_state=-1,
            result_name="layout",
        ),
    ]
    with foxes.Engine.new(backend, n_procs=2, chunk_size_states=2, verbosity=0):
        result = algorithm.calc_outputs(outputs=outputs, farm_results=farm_data)
    assert float(result["rose"]["frequency"].sum()) == pytest.approx(100.0)
    np.testing.assert_allclose(result["layout"][:, 0], farm_data[FV.X].values[-1])


@pytest.mark.parametrize(
    "parameters",
    [
        {"ws_bins": [0]},
        {"ws_bins": [0, 5, 5]},
        {"ws_bins": [0, np.nan, 10]},
        {"wd_sectors": 0},
        {"wd_sectors": 1.5},
    ],
)
def test_rose_rejects_invalid_histogram_configuration(algorithm, parameters):
    output = RosePlotOutput(plot=False, **parameters)
    with pytest.raises(ValueError, match="sectors and increasing bins"):
        output.initialize(algorithm)


@pytest.mark.parametrize("plane", ["xy", "xz", "yz"])
@pytest.mark.parametrize("quiver", [False, True])
def test_flow_plot_collects_vectors_and_geometry_only_when_used(
    algorithm, monkeypatch, plane, quiver
):
    import matplotlib.pyplot as plt
    from foxes.models.outputs import SliceData

    output = FlowPlots2D(
        var=FV.TI,
        plane=plane,
        grid_pars={"n_img_points": (2, 3)},
        plot_pars={"quiver_n": 1} if quiver else {},
    )
    calculate = SliceData.calculate
    inspected = []

    def capture(self, *args, **kwargs):
        result = calculate(self, *args, **kwargs)
        payload = result[self._result_key]
        assert not payload["farm"]
        assert set(payload["values"]) == ({FV.TI, FV.WD, FV.WS} if quiver else {FV.TI})
        assert all(value.shape[0] == 1 for value in payload["values"].values())
        inspected.append(True)
        return result

    monkeypatch.setattr(FlowPlots2D, "calculate", capture)
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=2, verbosity=0
    ):
        figure = algorithm.calc_farm(outputs=output)
    try:
        assert inspected
        assert len(figure.axes[0].collections) >= (2 if quiver else 1)
    finally:
        plt.close(figure)
