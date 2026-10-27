from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import foxes
import foxes.input.states.field_data as field_data_module
import foxes.variables as FV
from foxes.input.states import FieldData, LatLonFieldData
from foxes.input.states.dataset_states import DatasetStates
from foxes.output import FarmLayoutOutput
from foxes.utils.geom2d.polygon import ClosedPolygon


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
    ("stride", "error"),
    [
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
        (True, TypeError),
        (None, TypeError),
    ],
)
def test_grid_point_plot_stride_requires_positive_integer(stride, error):
    with pytest.raises(
        error, match="grid_point_plot_stride must be a positive integer"
    ):
        _new_states(FieldData, grid_point_plot_stride=stride)


@pytest.mark.parametrize("states_type", [FieldData, LatLonFieldData])
def test_grid_point_plot_farm_pars_are_copied_and_validated(states_type):
    overrides = {"title": "", "alpha": 0, "annotate": 0}
    states = _new_states(states_type, grid_point_plot_farm_pars=overrides)
    assert states.grid_point_plot_farm_pars == overrides
    assert states.grid_point_plot_farm_pars is not overrides
    assert _new_states(states_type).grid_point_plot_farm_pars == {}
    with pytest.raises(
        TypeError, match="grid_point_plot_farm_pars must be a dictionary"
    ):
        _new_states(states_type, grid_point_plot_farm_pars=[])


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
@pytest.mark.parametrize("farm_pars", [None, {"title": "", "alpha": 0, "annotate": 0}])
def test_grid_point_plot_pars_are_forwarded(
    monkeypatch, tmp_path, states_type, data, expected_zorder, farm_pars
):
    states = _new_states(
        states_type,
        grid_point_plot=str(tmp_path / "grid_points.png"),
        grid_point_plot_pars={"color": "darkblue", "alpha": 1.0},
        grid_point_plot_farm_pars=farm_pars,
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
    layout_pars = {"fig": fig, "ax": ax, "annotate": 0, "fontsize": 12}
    if states_type is FieldData:
        layout_pars["zorder"] = 10
    layout_pars.update(farm_pars or {})
    assert (
        field_data_module.FarmLayoutOutput.return_value.get_figure.call_args.kwargs
        == layout_pars
    )
    fig.savefig.assert_called_once()


@pytest.mark.parametrize("plot_stride", [1, 15, 2000])
def test_dense_grid_points_have_small_markers_and_visible_boundary(
    monkeypatch, tmp_path, plot_stride
):
    plot_file = tmp_path / "cfd_grid_points.png"
    boundary = ClosedPolygon(
        np.array([[10000.0, 10000.0], [30000.0, 10000.0], [20000.0, 40000.0]])
    )
    farm = foxes.WindFarm()
    farm.add_turbine(foxes.Turbine([20000.0, 20000.0], H=100.0, D=182.0), verbosity=0)
    states = _new_states(
        FieldData,
        grid_point_plot=plot_file,
        grid_point_plot_pars={"markersize": 2.0, "markeredgewidth": 0.0, "alpha": 0.6},
        grid_point_plot_stride=plot_stride,
        grid_point_plot_farm_pars={
            "title": "",
            "alpha": 0.0,
            "annotate": 0,
            "bargs": {
                "boundary": boundary,
                "show_boundary": True,
                "fill_mode": None,
                "pars_boundary": {
                    "edgecolor": "black",
                    "facecolor": "none",
                    "linewidth": 1.5,
                    "zorder": 20,
                },
            },
        },
    )
    coordinates = np.linspace(0.0, 80000.0, 1661)
    data = xr.Dataset(coords={"UTMX": coordinates, "UTMY": coordinates})
    close_figure = plt.close
    monkeypatch.setattr(DatasetStates, "preproc_first", Mock())
    monkeypatch.setattr(field_data_module.plt, "close", Mock())
    states.preproc_first(SimpleNamespace(farm=farm), data)
    figure = plt.gcf()
    try:
        axis = figure.axes[0]
        points, outline = axis.lines[0], axis.patches[-1]
        assert points.get_markersize() == 2.0
        assert points.get_markeredgewidth() == 0.0
        expected_axis = np.unique(
            np.concatenate((coordinates[::plot_stride], coordinates[-1:]))
        )
        assert (
            sum(len(line.get_xdata()) for line in axis.lines) == len(expected_axis) ** 2
        )
        assert points.get_ydata().max() == coordinates[-1]
        assert axis.lines[-1].get_xdata().max() == coordinates[-1]
        np.testing.assert_array_equal(data["UTMX"], coordinates)
        np.testing.assert_array_equal(data["UTMY"], coordinates)
        np.testing.assert_allclose(outline.get_edgecolor(), [0.0, 0.0, 0.0, 1.0])
        assert outline.get_facecolor()[-1] == 0.0
        assert outline.get_zorder() > points.get_zorder()
        assert outline.get_visible()
        assert axis.get_title() == ""
        assert axis.collections[0].get_alpha() == 0.0
        assert farm.boundary is None
        assert plot_file.stat().st_size > 0
        print(f"Synthetic CFD plot: {plot_file}")
    finally:
        close_figure(figure)


@pytest.mark.parametrize("mode", ["default", "override", "disabled"])
def test_farm_boundary_override_changes_only_the_figure(mode):
    points = np.array([[0.0, 0.0], [1000.0, 0.0], [0.0, 1000.0]])
    original = Mock(wraps=ClosedPolygon(points))
    replacement = Mock(wraps=ClosedPolygon(points + 2000.0))
    farm = foxes.WindFarm(boundary=original)
    farm.add_turbine(foxes.Turbine([100.0, 100.0], H=100.0, D=182.0), verbosity=0)
    bounds = farm.get_xy_bounds()
    bargs = {"show_boundary": True, "fill_mode": None}
    if mode != "default":
        bargs["boundary"] = replacement if mode == "override" else None
    expected_bargs = bargs.copy()
    figure, axis = plt.subplots()
    try:
        FarmLayoutOutput(farm).get_figure(fig=figure, ax=axis, bargs=bargs)
        assert len(axis.patches) == (0 if mode == "disabled" else 1)
        assert original.add_to_figure.call_count == (1 if mode == "default" else 0)
        assert replacement.add_to_figure.call_count == (1 if mode == "override" else 0)
        assert farm.boundary is original
        np.testing.assert_array_equal(farm.get_xy_bounds(), bounds)
        assert bargs == expected_bargs
    finally:
        plt.close(figure)
