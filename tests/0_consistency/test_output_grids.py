from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.core import FData, MData
from foxes.output import grids


@pytest.fixture
def farm_results():
    return xr.Dataset(
        {
            FV.X: ((FC.STATE, FC.TURBINE), [[0.0], [10.0], [20.0]]),
            FV.Y: ((FC.STATE, FC.TURBINE), [[0.0], [20.0], [40.0]]),
            FV.H: ((FC.STATE, FC.TURBINE), [[80.0], [90.0], [100.0]]),
        },
        coords={FC.STATE: [10, 20, 30]},
    )


@pytest.mark.parametrize("orientation", ["xy", "xz", "yz"])
def test_grid_coordinates_are_static(farm_results, orientation):
    grid_function = getattr(grids, f"get_grid_{orientation}")
    positions = grid_function(farm_results, n_img_points=(2, 3))
    grid_points = positions[-1]
    assert grid_points.shape == (6, 3)
    assert grid_points.nbytes == 6 * 3 * 8
    if orientation == "xy":
        expected = np.array(
            [
                [first, second, positions[2]]
                for first in positions[0]
                for second in positions[1]
            ]
        )
    elif orientation == "xz":
        expected = np.array(
            [
                [first, positions[1], second]
                for first in positions[0]
                for second in positions[2]
            ]
        )
    else:
        expected = np.array(
            [
                [positions[0], first, second]
                for first in positions[1]
                for second in positions[2]
            ]
        )
    np.testing.assert_allclose(grid_points, expected, atol=1e-12)


def test_xy_grid_memory_does_not_scale_with_states(monkeypatch):
    original_zeros = np.zeros

    def static_zeros(shape, **kwargs):
        assert len(shape) == 3
        return original_zeros(shape, **kwargs)

    monkeypatch.setattr(grids.np, "zeros", static_zeros)
    grid_points = grids.get_grid_xy(
        SimpleNamespace(sizes={FC.STATE: 1090512}),
        n_img_points=(216, 300),
        xmin=0.0,
        xmax=215.0,
        ymin=0.0,
        ymax=299.0,
        z=100.0,
    )[-1]
    assert grid_points.shape == (64800, 3)
    assert grid_points.nbytes == 1555200


@pytest.mark.parametrize("orientation", ["xy", "xz", "yz"])
def test_grid_selection_conflict_is_rejected(farm_results, orientation):
    with pytest.raises(ValueError, match="Choose either"):
        getattr(grids, f"get_grid_{orientation}")(
            farm_results,
            n_img_points=(2, 3),
            states_sel=[10],
            states_isel=[0],
        )


def test_selected_states_determine_xy_bounds(farm_results):
    x_pos, y_pos, z_pos, grid_points = grids.get_grid_xy(
        farm_results,
        n_img_points=(2, 3),
        states_isel=[1],
        xspace=1.0,
        yspace=2.0,
    )
    np.testing.assert_allclose(x_pos, [9.0, 11.0])
    np.testing.assert_allclose(y_pos, [18.0, 20.0, 22.0])
    assert z_pos == 90.0
    assert grid_points.shape == (6, 3)


@pytest.fixture
def flow_algorithm():
    states = foxes.input.states.StatesTable(
        data_source=pd.DataFrame(
            {
                FV.WS: [6.0, 7.0, 8.0, 9.0, 10.0],
                FV.WEIGHT: [0.1, 0.2, 0.3, 0.15, 0.25],
            },
            index=[10, 20, 30, 40, 50],
        ),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.WD: 270.0, FV.TI: 0.06, FV.RHO: 1.225},
    )
    farm = foxes.WindFarm()
    farm.add_turbine(
        foxes.Turbine(xy=[0.0, 0.0], H=100.0, D=100.0, turbine_models=["null_type"]),
        verbosity=0,
    )
    return foxes.algorithms.Downwind(
        farm, states, rotor_model="centre", wake_models=[], verbosity=0
    )


@pytest.mark.parametrize("orientation", ["xy", "xz", "yz"])
@pytest.mark.parametrize(
    "selection",
    [None, {"states_sel": [40, 20]}, {"states_isel": [3, 1]}, {"states_sel": [30]}],
)
def test_flow_slices_use_static_points_across_chunks(
    flow_algorithm, orientation, selection
):
    selection = {} if selection is None else selection
    all_states = np.array([10, 20, 30, 40, 50])
    selected_states = selection.get(
        "states_sel", all_states[selection.get("states_isel", slice(None))]
    )
    selected_indices = (np.asarray(selected_states) - 10) // 10
    expected_speeds = np.array([6.0, 7.0, 8.0, 9.0, 10.0])[selected_indices]
    expected_weights = np.array([0.1, 0.2, 0.3, 0.15, 0.25])[selected_indices]
    expected_mean = np.dot(expected_weights, expected_speeds)
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=2, verbosity=0
    ):
        farm_data = flow_algorithm.calc_farm(finalize=False)
        output = foxes.output.FlowPlots2D(flow_algorithm, farm_data)
        mean_output = getattr(output, f"get_mean_data_{orientation}")(
            FV.WS, n_img_points=(2, 3), **selection
        )
        state_output = getattr(output, f"get_states_data_{orientation}")(
            FV.WS, n_img_points=(2, 3), **selection
        )
    assert mean_output[-1][-1].shape == (6, 3)
    mean_ws_index = mean_output[0]["variables"].index(FV.WS)
    np.testing.assert_allclose(mean_output[1][:, :, mean_ws_index], expected_mean)
    np.testing.assert_array_equal(state_output[2], selected_states)
    assert state_output[-1][-1].shape == (6, 3)
    state_ws_index = state_output[0]["variables"].index(FV.WS)
    np.testing.assert_allclose(
        state_output[1][:, :, :, state_ws_index],
        np.broadcast_to(
            np.array(expected_speeds)[:, None, None], (len(selected_states), 2, 3)
        ),
    )


def test_precalc_grid_broadcasts_only_the_active_chunk(farm_results):
    point_model = Mock()

    def calculate(algo, mdata, fdata, tdata, **kwargs):
        targets = tdata[FC.TARGETS]
        assert targets.shape == (2, 6, 1, 3)
        assert targets.strides[0] == 0
        return {FV.WS: np.full((2, 6), 8.0)}

    point_model.calculate.side_effect = calculate
    algo = SimpleNamespace(_collect_point_models=lambda: (point_model, [{}]))
    output = foxes.output.FlowPlots2D(algo, farm_results)
    mdata = MData(data={FC.STATE: np.array([10, 20])}, dims={FC.STATE: (FC.STATE,)})
    data, states, grid_data = output.precalc_chunk_xy(
        FV.WS,
        mdata,
        FData.from_sizes(n_states=2, n_turbines=1),
        resolution=100.0,
        xmin=0.0,
        xmax=100.0,
        ymin=0.0,
        ymax=200.0,
        z=100.0,
    )
    assert grid_data[-1].shape == (6, 3)
    np.testing.assert_array_equal(states, [10, 20])
    np.testing.assert_allclose(data, 8.0)
