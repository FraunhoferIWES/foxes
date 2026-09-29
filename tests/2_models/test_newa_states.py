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
