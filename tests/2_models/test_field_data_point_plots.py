from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import xarray as xr

import foxes.input.states.field_data as field_data_module
import foxes.variables as FV
from foxes.input.states import FieldData, LatLonFieldData
from foxes.input.states.dataset_states import DatasetStates


def _new_states(states_type, **kwargs):
    if states_type is LatLonFieldData:
        kwargs["utm_zone"] = "31U"
    return states_type("unused.nc", output_vars=[FV.WS], **kwargs)


@pytest.mark.parametrize(
    ("states_type", "expected"),
    [
        (
            FieldData,
            {
                "color": "blue",
                "alpha": 0.2,
                "marker": ".",
                "linestyle": "None",
                "zorder": 5,
            },
        ),
        (
            LatLonFieldData,
            {
                "color": "blue",
                "alpha": 0.2,
                "marker": ".",
                "linestyle": "None",
            },
        ),
    ],
)
@pytest.mark.parametrize("plot_pars", [None, {}])
def test_grid_point_plot_pars_defaults(states_type, expected, plot_pars):
    states = _new_states(states_type, grid_point_plot_pars=plot_pars)

    assert states.grid_point_plot_pars == expected


@pytest.mark.parametrize("states_type", [FieldData, LatLonFieldData])
def test_grid_point_plot_pars_override_defaults_without_mutating_input(states_type):
    plot_pars = {"c": "darkblue", "alpha": 1.0}

    states = _new_states(states_type, grid_point_plot_pars=plot_pars)

    assert states.grid_point_plot_pars["color"] == "darkblue"
    assert states.grid_point_plot_pars["alpha"] == 1.0
    assert states.grid_point_plot_pars["marker"] == "."
    assert states.grid_point_plot_pars["linestyle"] == "None"
    assert plot_pars == {"c": "darkblue", "alpha": 1.0}


def test_lat_lon_field_data_preserves_positional_utm_zone():
    states = LatLonFieldData(
        "unused.nc",
        "Time",
        FV.LAT,
        FV.LON,
        None,
        None,
        None,
        "31U",
        output_vars=[FV.WS],
    )

    assert states._LatLonFieldData__utm_zone == "31U"


@pytest.mark.parametrize("states_type", [FieldData, LatLonFieldData])
def test_grid_point_plot_pars_rejects_non_dictionary(states_type):
    with pytest.raises(TypeError, match="grid_point_plot_pars must be a dictionary"):
        _new_states(states_type, grid_point_plot_pars=[])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("states_type", "data", "expected_zorder"),
    [
        (
            FieldData,
            xr.Dataset(coords={"UTMX": [0.0, 1.0], "UTMY": [2.0, 3.0]}),
            5,
        ),
        (
            LatLonFieldData,
            xr.Dataset(coords={FV.LON: [8.0, 8.1], FV.LAT: [53.0, 53.1]}),
            None,
        ),
    ],
)
def test_grid_point_plot_pars_are_forwarded(
    monkeypatch, tmp_path, states_type, data, expected_zorder
):
    states = _new_states(
        states_type,
        grid_point_plot=str(tmp_path / "grid_points.png"),
        grid_point_plot_pars={"color": "darkblue", "alpha": 1.0},
    )
    algo = SimpleNamespace(farm=SimpleNamespace(wind_farm_names=["farm"]))
    fig = Mock()
    ax = Mock()
    monkeypatch.setattr(DatasetStates, "preproc_first", Mock())
    monkeypatch.setattr(
        field_data_module,
        "config",
        SimpleNamespace(utm_zone_set=True, utm_zone=(31, "U")),
    )
    monkeypatch.setattr(field_data_module, "from_lonlat", lambda values: values)
    monkeypatch.setattr(field_data_module.plt, "subplots", Mock(return_value=(fig, ax)))
    monkeypatch.setattr(field_data_module.plt, "close", Mock())
    monkeypatch.setattr(field_data_module, "FarmLayoutOutput", Mock())

    states.preproc_first(algo, data)

    expected = {
        "color": "darkblue",
        "alpha": 1.0,
        "marker": ".",
        "linestyle": "None",
    }
    if expected_zorder is not None:
        expected["zorder"] = expected_zorder
    assert ax.plot.call_args.kwargs == expected
    fig.savefig.assert_called_once()
