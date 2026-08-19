import numpy as np
import pandas as pd
import xarray as xr
import foxes

from foxes.core import MData, TData
from foxes.input.states.dataset_states import DatasetStates
from foxes.input.states.point_cloud_data import PointCloudData, TurbinePointCloud
import foxes.constants as FC
import foxes.variables as FV


class _AlgoMock:
    pass


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
