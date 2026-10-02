from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import foxes.input.states.newa_states as newa_states_module
from foxes.core import MData
from foxes.input.states import NEWAStates
from foxes.input.states.dataset_states import DatasetStates
import foxes.variables as FV


def _interpolation_data():
    grid_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    values = np.array([[0.0], [1.0], [1.0]])
    return grid_points, values


@pytest.mark.parametrize("plot_pars", [None, {}])
def test_newa_wrf_point_plot_pars_defaults(plot_pars):
    states = NEWAStates("unused.nc", wrf_point_plot_pars=plot_pars)

    assert states.wrf_point_plot_pars == {
        "color": "blue",
        "alpha": 0.2,
        "marker": ".",
        "linestyle": "None",
    }


def test_newa_wrf_point_plot_pars_override_defaults_without_mutating_input():
    plot_pars = {"c": "darkblue", "alpha": 1.0}

    states = NEWAStates("unused.nc", wrf_point_plot_pars=plot_pars)

    assert states.wrf_point_plot_pars == {
        "color": "darkblue",
        "alpha": 1.0,
        "marker": ".",
        "linestyle": "None",
    }
    assert plot_pars == {"c": "darkblue", "alpha": 1.0}


def test_newa_forwards_wrf_point_plot_pars(monkeypatch, tmp_path):
    plot_file = tmp_path / "wrf_points.png"
    states = NEWAStates(
        "unused.nc",
        wrf_point_plot=plot_file,
        wrf_point_plot_pars={"color": "darkblue", "alpha": 1.0},
    )
    states._heights = [100.0]
    data = {
        "XLON": SimpleNamespace(values=np.array([[0.0, 1.0], [0.0, 1.0]])),
        "XLAT": SimpleNamespace(values=np.array([[0.0, 0.0], [1.0, 1.0]])),
    }
    algo = SimpleNamespace(farm=SimpleNamespace(wind_farm_names=["farm"]))
    fig = Mock()
    ax = Mock()
    monkeypatch.setattr(DatasetStates, "preproc_first", Mock())
    monkeypatch.setattr(newa_states_module, "from_lonlat", lambda values: values)
    monkeypatch.setattr(
        newa_states_module.plt, "subplots", Mock(return_value=(fig, ax))
    )
    monkeypatch.setattr(newa_states_module.plt, "close", Mock())
    monkeypatch.setattr(newa_states_module, "FarmLayoutOutput", Mock())

    states.preproc_first(algo, data)

    assert ax.plot.call_args.kwargs == {
        "color": "darkblue",
        "alpha": 1.0,
        "marker": ".",
        "linestyle": "None",
    }
    fig.savefig.assert_called_once_with(plot_file, bbox_inches="tight")


def test_newa_wrf_point_plot_pars_rejects_non_dictionary():
    with pytest.raises(TypeError, match="wrf_point_plot_pars must be a dictionary"):
        NEWAStates("unused.nc", wrf_point_plot_pars=[])  # type: ignore[arg-type]


def test_newa_interpolates_spatial_data_without_state_labels():
    states = NEWAStates("unused.nc", output_vars=[FV.WS])
    grid_points, values = _interpolation_data()

    result = states.interpolate_data(
        MData(),
        [FV.X, FV.Y],
        values,
        np.array([[0.25, 0.25]]),
        [FV.WS],
        state_labels=None,
        gpts=grid_points,
    )

    np.testing.assert_allclose(result, [[0.5]])


def test_newa_reports_spatial_interpolation_error_without_state_labels():
    states = NEWAStates("unused.nc", output_vars=[FV.WS])
    grid_points, values = _interpolation_data()

    with pytest.raises(ValueError, match="outside of bounds"):
        states.interpolate_data(
            MData(),
            [FV.X, FV.Y],
            values,
            np.array([[2.0, 2.0]]),
            [FV.WS],
            state_labels=None,
            gpts=grid_points,
        )


def test_newa_reports_state_data_error_without_state_labels():
    states = NEWAStates("unused.nc", output_vars=[FV.WS])
    grid_points, values = _interpolation_data()
    state_values = np.repeat(values[:, None, :], 2, axis=1)

    with pytest.raises(ValueError, match="outside of bounds"):
        states.interpolate_data(
            MData(),
            [FV.X, FV.Y],
            state_values,
            np.array([[2.0, 2.0]]),
            [FV.WS],
            state_labels=None,
            gpts=grid_points,
        )


def test_newa_interpolates_states_with_independent_valid_points():
    states = NEWAStates("unused.nc", output_vars=[FV.WS], check_input_nans=False)
    grid_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    values = np.array(
        [
            [[7.0], [10.0]],
            [[np.nan], [11.0]],
            [[7.0], [11.0]],
            [[7.0], [12.0]],
        ]
    )

    result = states.interpolate_data(
        MData(),
        [FV.X, FV.Y],
        values,
        np.array([[0.8, 0.1]]),
        [FV.WS],
        state_labels=np.array(["state-0", "state-1"]),
        gpts=grid_points,
    )

    np.testing.assert_allclose(result, [[[7.0], [10.9]]])


def test_newa_preserves_finite_boundary_values_between_states():
    states = NEWAStates("unused.nc", output_vars=[FV.WS], check_input_nans=False)
    grid_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    values = np.array(
        [
            [[4.0], [10.0]],
            [[5.0], [np.nan]],
            [[6.0], [11.0]],
            [[7.0], [12.0]],
        ]
    )

    result = states.interpolate_data(
        MData(),
        [FV.X, FV.Y],
        values,
        np.array([[1.0, 0.0]]),
        [FV.WS],
        state_labels=np.array(["finite", "missing"]),
        gpts=grid_points,
    )

    np.testing.assert_allclose(result[:, 0], [[5.0]])
    assert np.isfinite(result[:, 1]).all()


def test_newa_recovers_source_points_rejected_by_linear_griddata():
    states = NEWAStates("unused.nc", output_vars=[FV.WS])
    grid_points = np.array(
        [
            [527641.3981037099, 6975685.023524016],
            [527749.0381092395, 7003595.903495733],
            [527811.6986173344, 7019545.854586328],
            [531749.6079805237, 7001586.549207918],
            [533693.7101425079, 6985629.205895866],
            [535658.8440108196, 6975653.049863696],
            [563852.1682889046, 7009440.208922766],
            [563866.4632408092, 7013427.22964292],
            [565727.8676016734, 6975537.119269336],
            [565735.4604211745, 6977529.878374044],
            [565918.6191978771, 7025380.3368212925],
        ]
    )
    values = np.arange(len(grid_points), dtype=float)[:, None]

    result = states.interpolate_data(
        MData(),
        [FV.X, FV.Y],
        values,
        grid_points,
        [FV.WS],
        gpts=grid_points,
    )

    np.testing.assert_allclose(result, values)


def test_newa_does_not_fill_outside_original_convex_hull():
    states = NEWAStates("unused.nc", output_vars=[FV.WS], check_input_nans=False)
    grid_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.25, 0.25]])
    values = np.array([[0.0], [1.0], [1.0], [np.nan]])

    with pytest.raises(ValueError, match="Interpolation method 'linear' failed"):
        states.interpolate_data(
            MData(),
            [FV.X, FV.Y],
            values,
            np.array([[0.9, 0.9]]),
            [FV.WS],
            gpts=grid_points,
        )


def test_newa_fills_missing_source_hull_gap_in_three_dimensions():
    states = NEWAStates("unused.nc", output_vars=[FV.WS], check_input_nans=False)
    grid_points = np.indices((2, 2, 2)).reshape(3, -1).T.astype(float)
    values = np.full((len(grid_points), 1), 7.0)
    values[np.all(grid_points == [1.0, 0.0, 0.0], axis=1)] = np.nan

    result = states.interpolate_data(
        MData(),
        [FV.X, FV.Y, FV.H],
        values,
        np.array([[0.8, 0.1, 0.1]]),
        [FV.WS],
        gpts=grid_points,
    )

    np.testing.assert_allclose(result, [[7.0]])
