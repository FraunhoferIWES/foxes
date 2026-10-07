import numpy as np
import pandas as pd
import pytest
import xarray as xr
import foxes
from types import SimpleNamespace
from unittest.mock import Mock

import foxes.input.states.point_cloud_data as point_cloud_module
from foxes.core import MData, TData
from foxes.input.states import FieldData, MesoMicroField
from foxes.input.states.dataset_states import DatasetStates
from foxes.input.states.point_cloud_data import (
    PointCloudData,
    TurbinePointCloud,
    _interpolate_point_cloud_fields,
)
import foxes.constants as FC
import foxes.variables as FV


class _AlgoMock:
    pass


def _new_plot_states(states_type, **kwargs):
    if states_type == "BinnedPointCloudData":
        return foxes.input.states.BinnedPointCloudData("unused.nc", **kwargs)
    if states_type == "WeibullPointCloud":
        kwargs.update(wd_coord=FV.WD, ws_bins=[0.0, 10.0])
    if states_type != "TurbinePointCloud":
        kwargs.update(x_ncvar=FV.X, y_ncvar=FV.Y)
    return getattr(foxes.input.states, states_type)(
        data_source=xr.Dataset(), output_vars=[FV.WS, FV.WD], **kwargs
    )


@pytest.mark.parametrize(
    "states_type",
    [
        "PointCloudData",
        "WeibullPointCloud",
        "BinnedPointCloudData",
        "TurbinePointCloud",
    ],
)
def test_point_cloud_grid_point_plot_parameters(states_type):
    plot_pars = {"c": "darkblue", "alpha": 1.0}
    farm_pars = {"title": "", "alpha": 0, "annotate": 0}
    states = _new_plot_states(
        states_type,
        grid_point_plot_pars=plot_pars,
        grid_point_plot_farm_pars=farm_pars,
    )
    assert states.grid_point_plot is None
    assert states.grid_point_plot_pars == {
        "color": "darkblue",
        "alpha": 1.0,
        "marker": ".",
        "linestyle": "None",
        "zorder": 5,
    }
    assert states.grid_point_plot_farm_pars == farm_pars
    assert states.grid_point_plot_farm_pars is not farm_pars
    assert plot_pars == {"c": "darkblue", "alpha": 1.0}


@pytest.mark.parametrize(
    "states_type",
    [
        "PointCloudData",
        "WeibullPointCloud",
        "BinnedPointCloudData",
        "TurbinePointCloud",
    ],
)
@pytest.mark.parametrize(
    "parameter", ["grid_point_plot_pars", "grid_point_plot_farm_pars"]
)
def test_point_cloud_grid_point_plot_rejects_invalid_parameters(states_type, parameter):
    with pytest.raises(TypeError, match=f"{parameter} must be a dictionary"):
        _new_plot_states(states_type, **{parameter: []})


@pytest.fixture
def cloud_plot(monkeypatch):
    fig, ax = Mock(), Mock()
    monkeypatch.setattr(DatasetStates, "preproc_first", Mock())
    monkeypatch.setattr(
        point_cloud_module.plt, "subplots", Mock(return_value=(fig, ax))
    )
    monkeypatch.setattr(point_cloud_module.plt, "close", Mock())
    layout = Mock()
    monkeypatch.setattr(point_cloud_module, "FarmLayoutOutput", layout)
    algo = SimpleNamespace(farm=SimpleNamespace(wind_farm_names=["farm"]))
    return algo, fig, ax, layout


@pytest.mark.parametrize("load_mode", ["preload", "lazy", "fly"])
def test_point_cloud_grid_point_plot_selected_points(cloud_plot, tmp_path, load_mode):
    algo, fig, ax, layout = cloud_plot
    data = xr.Dataset(
        {FV.X: (FC.POINT, [100.0, 200.0]), FV.Y: (FC.POINT, [300.0, 400.0])},
        coords={FC.POINT: [10, 20]},
    )
    original = data.copy(deep=True)
    states = _new_plot_states(
        "PointCloudData",
        load_mode=load_mode,
        sel={FC.POINT: [20]},
        grid_point_plot=tmp_path / "cloud.png",
        grid_point_plot_farm_pars={"title": "", "alpha": 0, "annotate": 0},
    )
    states.preproc_first(algo, data)
    np.testing.assert_array_equal(ax.plot.call_args.args[0], [200.0])
    np.testing.assert_array_equal(ax.plot.call_args.args[1], [400.0])
    assert ax.plot.call_args.kwargs == states.grid_point_plot_pars
    assert layout.return_value.get_figure.call_args.kwargs == {
        "fig": fig,
        "ax": ax,
        "annotate": 0,
        "fontsize": 12,
        "zorder": 10,
        "title": "",
        "alpha": 0,
    }
    fig.savefig.assert_called_once()
    point_cloud_module.plt.close.assert_called_once_with(fig)
    xr.testing.assert_identical(data, original)


def test_point_cloud_grid_point_plot_closes_figure_on_write_error(cloud_plot, tmp_path):
    algo, fig, ax, layout = cloud_plot
    states = _new_plot_states("PointCloudData", grid_point_plot=tmp_path / "cloud.png")
    fig.savefig.side_effect = OSError("cannot write plot")
    data = xr.Dataset({FV.X: (FC.POINT, [1.0]), FV.Y: (FC.POINT, [2.0])})
    with pytest.raises(OSError, match="cannot write plot"):
        states.preproc_first(algo, data)
    point_cloud_module.plt.close.assert_called_once_with(fig)


def test_point_cloud_grid_point_plot_skips_empty_selection(cloud_plot, tmp_path):
    algo, fig, ax, layout = cloud_plot
    states = _new_plot_states(
        "PointCloudData", isel={FC.POINT: []}, grid_point_plot=tmp_path / "cloud.png"
    )
    data = xr.Dataset({FV.X: (FC.POINT, [1.0]), FV.Y: (FC.POINT, [2.0])})
    states.preproc_first(algo, data)
    fig.savefig.assert_not_called()


@pytest.mark.parametrize(
    "states_type", ["PointCloudData", "WeibullPointCloud", "BinnedPointCloudData"]
)
@pytest.mark.parametrize("quiet", [False, True])
def test_point_cloud_grid_point_plot_renders_default_and_quiet_overlays(
    monkeypatch, tmp_path, states_type, quiet
):
    from matplotlib.figure import Figure

    farm = foxes.WindFarm(name="Plot farm")
    farm.add_turbine(
        foxes.Turbine(xy=[0.0, 0.0], H=100.0, D=100.0, turbine_models=["null_type"]),
        verbosity=0,
    )
    states = _new_plot_states(
        states_type,
        grid_point_plot=tmp_path / "cloud.png",
        grid_point_plot_farm_pars={"title": "", "alpha": 0.0, "annotate": 0}
        if quiet
        else None,
    )
    data = xr.Dataset({FV.X: (FC.POINT, [10.0, 20.0]), FV.Y: (FC.POINT, [30.0, 40.0])})
    saved = []
    savefig = Figure.savefig

    def capture_savefig(figure, *args, **kwargs):
        saved.append(figure)
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_savefig)
    states.preproc_first(SimpleNamespace(farm=farm), data)
    assert (tmp_path / "cloud.png").stat().st_size > 0
    axis = saved[0].axes[0]
    assert axis.get_title() == ("" if quiet else "Plot farm")
    assert axis.collections[0].get_alpha() == (0.0 if quiet else None)
    assert axis.lines[0].get_alpha() == 0.2
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), [10.0, 20.0])


def test_turbine_point_cloud_grid_point_plot_uses_farm_locations(
    monkeypatch, cloud_plot, tmp_path
):
    algo, fig, ax, layout = cloud_plot
    algo.farm.turbines = [SimpleNamespace(xy=np.array([10.0, 20.0]))]
    monkeypatch.setattr(DatasetStates, "load_data", Mock())
    states = _new_plot_states(
        "TurbinePointCloud", grid_point_plot=tmp_path / "turbines.png"
    )
    states.load_data(algo, {"coords": {}, "data_vars": {}, "extra_data": {}})
    np.testing.assert_array_equal(ax.plot.call_args.args[0], [10.0])
    np.testing.assert_array_equal(ax.plot.call_args.args[1], [20.0])
    fig.savefig.assert_called_once()


def test_default_state_indices_are_reconstructed_not_serialized():
    states = DatasetStates(
        data_source=xr.Dataset(),
        output_vars=[],
        load_mode="preload",
    )
    loaded_data = {
        "coords": {FC.STATE: np.array([0, 1])},
        "data_vars": {},
        "extra_data": {},
    }

    states._N = 2
    states._inds = np.array([0, 1], dtype=np.int32)
    states._update_loaded_state_indices(loaded_data)
    assert FC.STATE not in loaded_data["coords"]

    states._inds = np.array([0, 30], dtype=np.int32)
    states._update_loaded_state_indices(loaded_data)
    assert np.array_equal(loaded_data["coords"][FC.STATE], np.array([0, 30]))


def test_point_cloud_preload_builds_multidimensional_coords():
    data_source = xr.Dataset(
        data_vars={
            FV.X: ((FC.POINT,), np.array([100.0, 200.0])),
            FV.Y: ((FC.POINT,), np.array([300.0, 400.0])),
            FV.WS: ((FC.STATE, FC.POINT), np.array([[8.0, 9.0], [10.0, 11.0]])),
            FV.WD: ((FC.STATE, FC.POINT), np.array([[270.0, 271.0], [272.0, 273.0]])),
        },
        coords={FC.STATE: np.array([0, 1], dtype=np.int32)},
    )
    states = PointCloudData(
        data_source=data_source,
        output_vars=[FV.WS, FV.WD],
        states_coord=FC.STATE,
        point_coord=FC.POINT,
        x_ncvar=FV.X,
        y_ncvar=FV.Y,
    )

    loaded_data = {"coords": {}, "data_vars": {}, "extra_data": {}}
    states.load_data(_AlgoMock(), loaded_data)

    mdata = xr.Dataset(coords=loaded_data["coords"], data_vars=loaded_data["data_vars"])
    point_coord = states.var(FC.POINT)
    axis_coord = states.var(FC.XYH)

    assert tuple(mdata.coords[point_coord].dims) == (point_coord, axis_coord)
    assert mdata.coords[point_coord].shape == (2, 2)
    assert list(mdata.coords[axis_coord].to_numpy()) == [FV.X, FV.Y]


def test_point_cloud_get_grid_points_selects_reference_height():
    states = PointCloudData(
        data_source=xr.Dataset(),
        output_vars=[FV.WS, FV.WD],
    )
    point_coord = states.var(FC.POINT)
    points = np.array([[0.0, 0.0, 80.0], [0.0, 0.0, 100.0], [100.0, 0.0, 100.0]])
    loaded_data = {
        "coords": {point_coord: ((FC.POINT, FC.XYH), points)},
        "data_vars": {},
        "extra_data": {},
    }

    highest = states.get_grid_points(loaded_data=loaded_data, all_heights=False)
    at_80m = states.get_grid_points(
        loaded_data=loaded_data,
        all_heights=False,
        height=80.0,
    )

    np.testing.assert_allclose(highest, points[1:])
    np.testing.assert_allclose(at_80m, points[:1])

    loaded_data["coords"][point_coord] = (
        (FC.POINT, FC.XY),
        points[:, :2].astype(int),
    )
    at_90_5m = states.get_grid_points(
        loaded_data=loaded_data,
        all_heights=False,
        height=90.5,
    )
    np.testing.assert_allclose(
        at_90_5m,
        np.column_stack((points[:, :2], np.full(len(points), 90.5))),
    )


def test_point_cloud_interpolate_falls_back_to_nearest_on_qhull_error():
    states = PointCloudData(
        data_source=xr.Dataset(),
        output_vars=[FV.WS, FV.WD],
    )

    support_points = np.array([[0.0, 0.0], [300.0, 0.0]])
    evaluation_points = np.array([[0.0, 0.0], [100.0, 0.0]])
    data = np.array([[8.0, 270.0], [9.0, 271.0]])

    out = states.interpolate_data(
        mdata={},
        idims=[FC.POINT],
        d=data,
        pts=evaluation_points,
        vrs=[FV.WS, FV.WD],
        gpts=support_points,
    )

    assert out.shape == (2, 2)
    assert np.allclose(out[0], np.array([8.0, 270.0]))
    assert np.allclose(out[1], np.array([8.0, 270.0]))


def test_point_cloud_recovers_source_points_rejected_by_linear_griddata():
    states = PointCloudData(data_source=xr.Dataset(), output_vars=[FV.WS])
    support_points = np.array(
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
    data = np.arange(len(support_points), dtype=float)[:, None]

    out = states.interpolate_data(
        mdata={},
        idims=[FC.POINT],
        d=data,
        pts=support_points,
        vrs=[FV.WS],
        gpts=support_points,
    )

    np.testing.assert_allclose(out, data)
    chunked = _interpolate_point_cloud_fields(
        support_points,
        data,
        support_points,
        {"method": "linear", "rescale": True},
        check_input_nans=True,
        chunk_size_points=3,
    )
    np.testing.assert_allclose(chunked, data)


def test_point_cloud_interpolates_states_with_independent_valid_points():
    states = PointCloudData(
        data_source=xr.Dataset(),
        output_vars=[FV.WS],
        check_input_nans=False,
    )
    support_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    data = np.array(
        [
            [[4.0], [10.0]],
            [[5.0], [np.nan]],
            [[6.0], [11.0]],
            [[7.0], [12.0]],
        ]
    )

    out = states.interpolate_data(
        mdata={},
        idims=[FC.POINT],
        d=data,
        pts=np.array([[1.0, 0.0]]),
        vrs=[FV.WS],
        gpts=support_points,
    )

    np.testing.assert_allclose(out[:, 0], [[5.0]])
    assert np.isfinite(out[:, 1]).all()


def _interpolate_point_cloud(interp_pars):
    states = PointCloudData(
        data_source=xr.Dataset(),
        output_vars=[FV.WS],
        interp_pars=interp_pars,
    )
    support_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    evaluation_points = np.array([[0.25, 0.25], [1.2, 0.1]])
    data = np.array([[0.0], [1.0], [2.0]])
    return states.interpolate_data(
        mdata={},
        idims=[FC.POINT],
        d=data,
        pts=evaluation_points,
        vrs=[FV.WS],
        gpts=support_points,
    )


def test_point_cloud_uses_default_none_fill_when_bounds_errors_disabled():
    out = _interpolate_point_cloud({"bounds_error": False})

    np.testing.assert_allclose(out, [[0.75], [1.0]])


def test_point_cloud_keeps_nan_fill_when_bounds_errors_disabled():
    out = _interpolate_point_cloud({"bounds_error": False, "fill_value": np.nan})

    np.testing.assert_allclose(out[0], [0.75])
    assert np.isnan(out[1, 0])


def test_point_cloud_raises_for_default_none_fill_when_bounds_errors_enabled():
    with pytest.raises(ValueError, match="set bounds_error=False"):
        _interpolate_point_cloud({"bounds_error": True})


def test_point_cloud_reports_inside_bounds_outside_support_hull(capsys):
    states = PointCloudData(
        data_source=xr.Dataset(),
        output_vars=[FV.WS],
        interp_pars={"bounds_error": True},
    )
    support_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="outside the support hull"):
        states.interpolate_data(
            mdata={},
            idims=[FC.POINT],
            d=np.array([[0.0], [1.0], [2.0]]),
            pts=np.array([[0.8, 0.8]]),
            vrs=[FV.WS],
            gpts=support_points,
        )

    diagnostic = capsys.readouterr().out
    assert "Inside coordinate bounds:  [ True  True]" in diagnostic
    assert "Inside support hull:        False" in diagnostic


def test_point_cloud_keeps_finite_fill_when_bounds_errors_disabled():
    out = _interpolate_point_cloud({"bounds_error": False, "fill_value": -1.0})

    np.testing.assert_allclose(out, [[0.75], [-1.0]])


def test_point_cloud_interpolate_removes_invariant_axis_for_planar_support():
    states = PointCloudData(
        data_source=xr.Dataset(),
        output_vars=[FV.WS],
    )
    support_points = np.array(
        [
            [0.0, 0.0, 90.0],
            [1.0, 0.0, 90.0],
            [0.0, 1.0, 90.0],
            [1.0, 1.0, 90.0],
        ]
    )
    data = support_points[:, :1].copy()

    out = states.interpolate_data(
        mdata={},
        idims=[FC.POINT],
        d=data,
        pts=np.array([[0.5, 0.5, 90.0]]),
        vrs=[FV.WS],
        gpts=support_points,
    )

    np.testing.assert_allclose(out, [[0.5]])


def test_turbine_point_cloud_does_not_require_xy_cmap_for_preproc():
    data_source = xr.Dataset(
        data_vars={
            FV.WS: (("time", "turbine"), np.array([[8.0, 9.0], [10.0, 11.0]])),
            FV.WD: (("time", "turbine"), np.array([[270.0, 271.0], [272.0, 273.0]])),
        },
        coords={"time": np.array([0, 1], dtype=np.int32), "turbine": np.array([0, 1])},
    )

    states = TurbinePointCloud(
        data_source=data_source,
        output_vars=[FV.WS, FV.WD],
        states_coord="time",
        turbine_coord="turbine",
    )

    states.preproc_first(
        _AlgoMock(),
        data=data_source,
        bounds_extra_space=states.bounds_extra_space,
        height_bounds=None,
        verbosity=0,
    )

    assert states.bounds_extra_space is None


def test_turbine_point_cloud_interpolate_falls_back_to_nearest_on_qhull_error():
    data_source = xr.Dataset(
        data_vars={
            FV.WS: (("time", "turbine"), np.array([[8.0, 9.0]])),
            FV.WD: (("time", "turbine"), np.array([[270.0, 271.0]])),
        },
        coords={"time": np.array([0], dtype=np.int32), "turbine": np.array([0, 1])},
    )

    states = TurbinePointCloud(
        data_source=data_source,
        output_vars=[FV.WS, FV.WD],
        states_coord="time",
        turbine_coord="turbine",
    )

    # This setup produces only two support points in a single chunk.
    # Linear interpolation cannot build a simplex and should fall back to nearest.
    idims = [FC.TURBINE]
    d = np.array([[[8.0, 270.0], [9.0, 271.0]]], dtype=float)
    vrs = [FV.WS, FV.WD]
    times = np.array([0], dtype=np.int32)

    mdata = {
        FC.TURBINE: np.array([[[0.0, 0.0, 90.0], [300.0, 0.0, 90.0]]], dtype=float),
    }
    out = states.interpolate_data(
        mdata,
        idims,
        d,
        np.array([[[0.0, 0.0, 90.0], [100.0, 0.0, 90.0]]], dtype=float),
        vrs,
        times,
    )

    assert out.shape == (1, 2, 2)
    assert np.allclose(out[0, 0], np.array([8.0, 270.0]))
    assert np.allclose(out[0, 1], np.array([8.0, 270.0]))


def test_turbine_point_cloud_uses_default_none_fill():
    data_source = xr.Dataset(
        data_vars={
            FV.WS: (
                ("time", "turbine"),
                np.array([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]),
            ),
            FV.WD: (
                ("time", "turbine"),
                np.full((2, 3), 270.0),
            ),
        },
        coords={"time": np.array([0, 1], dtype=np.int32), "turbine": np.arange(3)},
    )
    states = TurbinePointCloud(
        data_source=data_source,
        output_vars=[FV.WS, FV.WD],
        states_coord="time",
        turbine_coord="turbine",
        interp_pars={"bounds_error": False},
    )
    support_points = np.array(
        [
            [[0.0, 0.0, 90.0], [1.0, 0.0, 90.0], [0.0, 1.0, 90.0]],
            [[0.0, 0.0, 90.0], [1.0, 0.0, 90.0], [0.0, 1.0, 90.0]],
        ]
    )
    evaluation_points = np.array(
        [
            [[0.25, 0.25, 90.0], [1.2, 0.1, 90.0], [0.0, 0.0, 90.0]],
            [[0.25, 0.25, 90.0], [1.2, 0.1, 90.0], [0.0, 0.0, 90.0]],
        ]
    )
    data = np.array([[[0.0], [1.0], [2.0]], [[10.0], [11.0], [12.0]]])

    out = states.interpolate_data(
        mdata={},
        idims=[FC.TURBINE],
        d=data,
        pts=evaluation_points,
        vrs=[FV.WS],
        gpts=support_points,
    )

    np.testing.assert_allclose(
        out,
        [[[0.75], [1.0], [0.0]], [[10.75], [11.0], [10.0]]],
    )


def test_dataset_states_calculate_handles_turbine_dim_without_not_implemented():
    class DummyStates(DatasetStates):
        def __init__(self):
            super().__init__(
                data_source=xr.Dataset(),
                output_vars=[FV.WS, FV.WD],
                fixed_vars={},
                var2ncvar={},
                load_mode="preload",
            )
            self._N = 1
            self._inds = np.array([0], dtype=np.int32)
            self._cmap = {FC.STATE: FC.STATE}
            self.received_pts = None

        def _get_calc_data(self, mdata, fdata):
            d = np.array([[[8.0, 270.0], [9.0, 271.0]]], dtype=float)
            return {(FC.STATE, FC.TURBINE, "vars0"): ([FV.WS, FV.WD], d)}, None

        def interpolate_data(self, mdata, idims, d, pts, vrs, state_indices):
            self.received_pts = np.array(pts, copy=True)
            return d

    states = DummyStates()
    mdata = MData(
        data={
            FC.STATE: np.array([0], dtype=np.int32),
            FC.TURBINE: np.array([[0.0, 0.0, 90.0], [100.0, 0.0, 90.0]], dtype=float),
        },
        dims={
            FC.STATE: (FC.STATE,),
            FC.TURBINE: (FC.TURBINE, FC.XYH),
        },
        name="mdata_turbine",
    )
    tdata = TData.from_points(
        points=np.array([[[0.0, 0.0, 90.0], [100.0, 0.0, 90.0]]], dtype=float),
        variables=[FV.WS, FV.WD],
    )

    results = states.calculate(algo=None, mdata=mdata, fdata=None, tdata=tdata)

    assert states.received_pts.shape == (2, 3)
    assert np.allclose(
        states.received_pts, np.array([[0.0, 0.0, 90.0], [100.0, 0.0, 90.0]])
    )
    assert np.allclose(results[FV.WS][0, :, 0], np.array([8.0, 9.0]))


def _dataset_reconstruction_case(dimension, state_dependent, n_states):
    grid = np.array([0.0, 1000.0]) if dimension == FV.X else np.array([0.0, 200.0])
    wind_speed = 8.0 + 0.01 * grid
    field_dims = (dimension,)
    if state_dependent:
        wind_speed = wind_speed[None, :] + np.arange(n_states)[:, None]
        field_dims = (FC.STATE, dimension)
    data_source = xr.Dataset(
        coords={
            FC.STATE: np.arange(n_states),
            FV.X: np.array([0.0, 1000.0]),
            FV.Y: np.array([-100.0, 100.0]),
            dimension: grid,
        },
        data_vars={
            FV.WS: (field_dims, wind_speed),
            FV.RHO: (field_dims, 1.0 + 0.01 * wind_speed),
            FV.WD: ((FC.STATE,), np.resize([270.0, 90.0], n_states)),
        },
    )
    states = FieldData(
        data_source,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.TI: 0.06},
        states_coord=FC.STATE,
        x_coord=FV.X,
        y_coord=FV.Y,
        h_coord=FV.H if dimension == FV.H else None,
        time_format=None,
        bounds_extra_space=None,
        height_bounds=(0.0, 200.0),
    )
    farm = foxes.WindFarm()
    locations = np.array([0.0, 250.0, 750.0])
    heights = np.array([60.0, 90.0, 120.0])
    for location, height in zip(locations, heights):
        farm.add_turbine(
            foxes.Turbine(xy=[location, 0.0], H=height, turbine_models=["null_type"]),
            verbosity=0,
        )
    algo = foxes.algorithms.Downwind(farm, states, wake_models=[], verbosity=0)
    expected = np.broadcast_to(
        8.0 + 0.01 * (locations if dimension == FV.X else heights), (n_states, 3)
    ).copy()
    if state_dependent:
        expected += np.arange(n_states)[:, None]
    return algo, expected


def _guard_dataset_reconstruction(monkeypatch, states):
    selections = []
    interpolate_data = states.interpolate_data

    class ReconstructionArray(np.ndarray):
        def __getitem__(self, indices):
            if (
                isinstance(indices, tuple)
                and len(indices) > 1
                and isinstance(indices[1], np.ndarray)
            ):
                assert indices[1].ndim == 2, "Cross-state point expansion"
                state_indices = indices[0]
                if isinstance(state_indices, (int, np.integer)) or (
                    isinstance(state_indices, np.ndarray)
                    and state_indices.shape == (self.shape[0], 1)
                ):
                    selections.append(indices[1].shape)
            return super().__getitem__(indices)

    def interpolate_with_guard(*args, **kwargs):
        return interpolate_data(*args, **kwargs).view(ReconstructionArray)

    monkeypatch.setattr(states, "interpolate_data", interpolate_with_guard)
    return selections


@pytest.mark.parametrize("dimension", [FV.X, FV.H])
@pytest.mark.parametrize("state_dependent", [False, True])
@pytest.mark.parametrize("n_states", [1, 3])
def test_dataset_states_reconstruct_downwind_order_without_cross_state_expansion(
    monkeypatch, dimension, state_dependent, n_states
):
    algo, expected = _dataset_reconstruction_case(dimension, state_dependent, n_states)
    selections = _guard_dataset_reconstruction(monkeypatch, algo.states)
    with foxes.Engine.new("single", verbosity=0):
        results = algo.calc_farm()

    assert results[FV.AMB_REWS].dims == (FC.STATE, FC.TURBINE)
    np.testing.assert_allclose(results[FV.AMB_REWS], expected)
    np.testing.assert_allclose(results[FV.AMB_RHO], 1.0 + 0.01 * expected)
    if n_states > 1:
        assert selections == [(n_states, 3)]
        assert not np.array_equal(results[FV.ORDER][0], results[FV.ORDER][1])
    else:
        assert not selections


@pytest.mark.parametrize("dimension", [FV.X, FV.H])
def test_dataset_states_reconstruction_preserves_bounds_errors(dimension):
    algo, _ = _dataset_reconstruction_case(dimension, True, 3)
    if dimension == FV.X:
        algo.farm.turbines[0].xy = np.array([2000.0, 0.0])
    else:
        algo.farm.turbines[0].H = 250.0
    with foxes.Engine.new("single", verbosity=0):
        with pytest.raises(ValueError, match="out of bounds"):
            algo.calc_farm()


@pytest.mark.parametrize("states_type", ["dataset", "meso-micro"])
@pytest.mark.parametrize("engine_type", ["single", "process"])
def test_layout_optimization_population_matches_individuals(states_type, engine_type):
    layout_module = pytest.importorskip("foxes_opt.problems.layout")
    objectives_module = pytest.importorskip("foxes_opt.objectives")
    algo, _ = _dataset_reconstruction_case(FV.X, True, 2)
    if states_type == "meso-micro":
        meso_algo, _ = _dataset_reconstruction_case(FV.X, True, 2)
        meso_algo.states.name = "meso_field"
        algo.states = MesoMicroField(
            micro_states=algo.states,
            meso_states=meso_algo.states,
            ref_points=[[0.0, 0.0, 90.0]],
        )
    algo.farm.boundary = foxes.utils.geom2d.Circle([500.0, 0.0], 1000.0)
    problem = layout_module.FarmLayoutOptProblem("layout_reconstruction", algo)
    objective = objectives_module.FarmVarObjective(
        problem, "ambient_rews", FV.AMB_REWS, "weights", "sum", False
    )
    problem.add_objective(objective)
    with foxes.Engine.new(engine_type, n_procs=2, chunk_size_states=3, verbosity=0):
        problem.initialize(verbosity=0)
        layouts = np.repeat(problem.initial_values_float()[None, :], 3, axis=0)
        layouts[1, ::2] += 100.0
        layouts[2, ::2] = [650.0, 50.0, 450.0]
        for candidates in (layouts, layouts[::-1]):
            vars_int = np.zeros((len(candidates), 0), dtype=int)
            population = problem.apply_population(vars_int, candidates)
            population_objectives = objective.calc_population(
                vars_int, candidates, population
            )
            for candidate_i, candidate in enumerate(candidates):
                individual = problem.apply_individual(vars_int[candidate_i], candidate)
                for variable in (FV.AMB_REWS, FV.AMB_RHO):
                    np.testing.assert_allclose(
                        population[variable].to_numpy().reshape(3, 2, 3)[candidate_i],
                        individual[variable],
                    )
                np.testing.assert_allclose(
                    population_objectives[candidate_i],
                    objective.calc_individual(
                        vars_int[candidate_i], candidate, individual
                    ),
                )
                assert all(turbine.xy.shape == (2,) for turbine in algo.farm.turbines)


def _run_ambient_rews(
    ws: np.ndarray,
    wd: np.ndarray,
    turbine_xy: np.ndarray,
    hubs: np.ndarray,
) -> np.ndarray:
    sdata = xr.Dataset(
        coords={
            FC.STATE: pd.date_range("2000-01-01", periods=ws.shape[0], freq="1h"),
            FC.TURBINE: np.arange(ws.shape[1]),
        },
        data_vars={
            "ws": ((FC.STATE, FC.TURBINE), ws),
            "wd": ((FC.STATE, FC.TURBINE), wd),
        },
    )

    states = TurbinePointCloud(
        data_source=sdata,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2ncvar={FV.WS: "ws", FV.WD: "wd"},
        fixed_vars={FV.TI: 0.06, FV.RHO: 1.225},
    )
    farm = foxes.WindFarm()
    for i, h in enumerate(hubs):
        farm.add_turbine(
            foxes.Turbine(
                xy=turbine_xy[i],
                H=float(h),
                turbine_models=["null_type"],
            ),
            verbosity=0,
        )

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        wake_models=[],
        rotor_model="centre",
        mbook=foxes.models.ModelBook(),
        verbosity=0,
    )
    with foxes.core.Engine.new("single", verbosity=0):
        return algo.calc_farm()[FV.AMB_REWS].to_numpy()


def test_turbine_point_cloud_preserves_values_for_mixed_hub_heights():
    ws = np.array([[10.0, 10.5, 9.5]] * 2)
    wd = np.full_like(ws, 270.0)
    turbine_xy = np.array([[0.0, 0.0], [500.0, 0.0], [1000.0, 0.0]])

    for hubs in (
        np.array([100.0, 100.0, 100.0]),
        np.array([100.0, 100.0, 90.0]),
        np.array([120.0, 100.0, 90.0]),
    ):
        got = _run_ambient_rews(ws, wd, turbine_xy, hubs)
        assert np.allclose(got, ws)


def test_turbine_point_cloud_preserves_values_for_grid_layout_with_varying_wd():
    rng = np.random.default_rng(3)
    n_states = 40
    n_turb = 6
    ws = rng.uniform(7.0, 12.0, size=(n_states, n_turb))
    wd = rng.uniform(240.0, 300.0, size=(n_states, 1))
    wd = np.repeat(wd, n_turb, axis=1)
    turbine_xy = np.array(
        [
            [0.0, 0.0],
            [500.0, 0.0],
            [1000.0, 0.0],
            [0.0, 500.0],
            [500.0, 500.0],
            [1000.0, 500.0],
        ]
    )
    hubs = np.full(n_turb, 100.0)

    got = _run_ambient_rews(ws, wd, turbine_xy, hubs)
    assert np.allclose(got, ws)


@pytest.mark.parametrize("load_mode", ["preload", "lazy", "fly"])
def test_turbine_point_cloud_load_modes_preserve_timestamp_labels(tmp_path, load_mode):
    state_index = pd.date_range("2020-01-01", periods=4, freq="h")
    ws = np.arange(12, dtype=float).reshape(4, 3) + 7.0
    data_source = xr.Dataset(
        coords={FC.STATE: state_index, FC.TURBINE: np.arange(3)},
        data_vars={
            FV.WS: ((FC.STATE, FC.TURBINE), ws),
            FV.WD: ((FC.STATE, FC.TURBINE), np.full_like(ws, 270.0)),
        },
    )
    data_path = tmp_path / "turbine_point_cloud.nc"
    data_source.to_netcdf(data_path)

    states = TurbinePointCloud(
        data_source=data_path,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.TI: 0.06, FV.RHO: 1.225},
        load_mode=load_mode,
    )
    farm = foxes.WindFarm()
    for turbine_i in range(3):
        farm.add_turbine(
            foxes.Turbine(
                xy=[500.0 * turbine_i, 0.0],
                H=100.0,
                turbine_models=["null_type"],
            ),
            verbosity=0,
        )
    algo = foxes.algorithms.Downwind(
        farm,
        states,
        wake_models=[],
        rotor_model="centre",
        verbosity=0,
    )
    write_nc = (
        {
            "out_dir": tmp_path,
            "base_name": "results",
            "split": 2,
            "ret_data": True,
        }
        if load_mode == "preload"
        else None
    )

    with foxes.Engine.new("single", chunk_size_states=2, verbosity=0):
        results = algo.calc_farm(write_nc=write_nc)

    np.testing.assert_array_equal(results[FC.STATE].to_numpy(), state_index.to_numpy())
    np.testing.assert_allclose(results[FV.AMB_REWS].to_numpy(), ws)
    if write_nc is not None:
        for file_i, expected_labels in enumerate((state_index[:2], state_index[2:])):
            with xr.open_dataset(tmp_path / f"results_{file_i:06d}.nc") as split_data:
                np.testing.assert_array_equal(
                    split_data[FC.STATE].to_numpy(), expected_labels.to_numpy()
                )
