from __future__ import annotations

from importlib import import_module
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV


DATA = Path(foxes.__file__).resolve().parent / "data"
STATES_DATA = DATA / "states"
POWER_DATA = DATA / "power_ct_curves"
FARM_DATA = DATA / "farms"
MODEL_DATA = DATA / "model_data"

NREL5MW = POWER_DATA / "NREL-5MW-D126-H90.csv"
TEST_FARM = FARM_DATA / "test_farm_67.csv"
TIMESERIES_100 = STATES_DATA / "timeseries_100.csv.gz"
TIMESERIES_3000 = STATES_DATA / "timeseries_3000.csv.gz"
WIND_ROSE = STATES_DATA / "wind_rose_bremen.csv"
WEIBULL_SECTORS = STATES_DATA / "weibull_sectors_12.csv"
WEIBULL_GRID = STATES_DATA / "weibull_grid.nc"
WEIBULL_CLOUD = STATES_DATA / "weibull_cloud_4.nc"
POINT_CLOUD = STATES_DATA / "point_cloud_100.nc"
WRF_TIMESERIES = STATES_DATA / "WRF-Timeseries-3000.nc"
WIND_ROTATION = STATES_DATA / "wind_rotation.nc"
ICON_HEIGHTS_A1 = MODEL_DATA / "icon_heights_A1.csv"
ICON_HEIGHTS_A2 = MODEL_DATA / "icon_heights_A2.csv"


UNSUPPORTED = {
    "foxes.input.states.dataset_states:DatasetStates": "DatasetStates is a generic data-loading base that requires subclass-specific coordinate mapping for realistic calc_farm integration.",
    "foxes.models.farm_models.turbine2farm:Turbine2FarmModel": "Farm model bridge is not collected by Downwind.calc_farm or calc_points directly.",
    "foxes.models.wake_models.wind.bastankhah16:Bastankhah2016Model": "Helper model is used internally by Bastankhah2016 and is not pluggable as a top-level algo model.",
}


def _load_class(model_path: str):
    module_name, class_name = model_path.split(":")
    return getattr(import_module(module_name), class_name)


def _alias(name: str) -> str:
    return f"smoke_{name.lower()}"


def _engine(engine_type="threads"):
    return foxes.Engine.new(engine_type=engine_type)


def _farm(turbine_models, n_turbines=2, H=90.0, xy_base=None, xy_step=None):
    farm = foxes.WindFarm()
    if xy_base is None:
        xy_base = [0.0, 0.0]
    if xy_step is None:
        xy_step = [600.0, 0.0]
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=xy_base,
        xy_step=xy_step,
        n_turbines=n_turbines,
        turbine_models=turbine_models,
        H=H,
        verbosity=0,
    )
    return farm


def _mbook_with_ttype(ttype=None, alias=None):
    mbook = foxes.models.ModelBook()
    if ttype is None:
        ttype = foxes.models.turbine_types.PCtFile(str(NREL5MW), rho=1.225)
    if alias is None:
        alias = _alias(type(ttype).__name__)
    if getattr(ttype, "D", None) is None:
        ttype.D = 126.0
    if getattr(ttype, "H", None) is None:
        ttype.H = 90.0
    if hasattr(ttype, "P_nominal") and getattr(ttype, "P_nominal", None) is None:
        ttype.P_nominal = 5000.0
    ttype.name = alias
    mbook.turbine_types[alias] = ttype
    return mbook, alias


def _assert_farm_results(farm_results):
    assert FC.STATE in farm_results.sizes
    assert FC.TURBINE in farm_results.sizes
    assert farm_results.sizes[FC.STATE] >= 1
    assert farm_results.sizes[FC.TURBINE] >= 1
    assert FV.P in farm_results


def _assert_point_results(point_results):
    assert FC.STATE in point_results.sizes
    assert FC.POINT in point_results.sizes
    assert point_results.sizes[FC.STATE] >= 1
    assert point_results.sizes[FC.POINT] >= 1
    assert FV.WS in point_results


def _write_newa_dataset(tmp_path: Path) -> Path:
    fpath = tmp_path / "newa_states.nc"
    times = pd.date_range("2023-01-01", periods=2, freq="10min").to_numpy()
    west_east = np.array([0.0, 1.0], dtype=float)
    south_north = np.array([0.0, 1.0], dtype=float)
    height = np.array([20.0, 90.0, 180.0], dtype=float)
    xlon = np.array([[8.00, 8.01], [8.00, 8.01]], dtype=float)
    xlat = np.array([[53.00, 53.00], [53.01, 53.01]], dtype=float)
    data = xr.Dataset(
        data_vars={
            "WS": (
                ("time", "height", "south_north", "west_east"),
                np.full((2, 3, 2, 2), 8.0),
            ),
            "WD": (
                ("time", "height", "south_north", "west_east"),
                np.full((2, 3, 2, 2), 270.0),
            ),
            "XLAT": (("south_north", "west_east"), xlat),
            "XLON": (("south_north", "west_east"), xlon),
        },
        coords={
            "time": times,
            "height": height,
            "west_east": west_east,
            "south_north": south_north,
        },
    )
    data.to_netcdf(fpath, engine=foxes.config.nc_engine)
    return fpath


def _write_icon_dataset(tmp_path: Path) -> Path:
    fpath = tmp_path / "icon_states.nc"
    times = pd.date_range("2023-01-01", periods=2, freq="10min").to_numpy()
    lons = np.array([8.00, 8.01], dtype=float)
    lats = np.array([53.00, 53.01], dtype=float)
    hmap = pd.read_csv(ICON_HEIGHTS_A2)
    hidx = np.array(
        [
            int((hmap["height"] - 20.0).abs().idxmin()),
            int((hmap["height"] - 90.0).abs().idxmin()),
            int((hmap["height"] - 180.0).abs().idxmin()),
        ],
        dtype=int,
    )
    data = xr.Dataset(
        data_vars={
            "U": (("time", "height", "lat", "lon"), np.full((2, 3, 2, 2), 8.0)),
            "V": (("time", "height", "lat", "lon"), np.zeros((2, 3, 2, 2))),
        },
        coords={
            "time": times,
            "height": hidx,
            "height_2": hidx,
            "lat": lats,
            "lon": lons,
        },
    )
    data.to_netcdf(fpath, engine=foxes.config.nc_engine)
    return fpath


def _single_state(profiles=None, **profdata):
    return foxes.input.states.SingleStateStates(
        wd=270.0,
        ti=0.08,
        rho=1.225,
        ws=8.0 if profiles is None else None,
        profiles={} if profiles is None else profiles,
        **profdata,
    )


def _timeseries_states(cls):
    if cls.__name__ == "TabStates":
        return cls(str(STATES_DATA / "winds100.tab"), output_vars=[FV.WS, FV.WD])
    if cls.__name__ == "WeibullSectors":
        return cls(
            str(WEIBULL_SECTORS),
            output_vars=[FV.WS, FV.WD, FV.TI],
            ws_bins=np.array([0.0, 5.0, 10.0, 15.0, 20.0]),
            var2ncvar={
                FV.WD: "wind_direction",
                FV.WEIGHT: "sector_probability",
                FV.WEIBULL_A: "weibull_a",
                FV.WEIBULL_k: "weibull_k",
                FV.TI: "turbulence_intensity",
            },
        )
    if cls.__name__ == "StatesTable":
        return cls(
            data_source=str(TIMESERIES_100),
            output_vars=[FV.WS, FV.WD],
            var2col={FV.WS: "ws", FV.WD: "wd"},
            fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
        )
    if cls.__name__ == "Timeseries":
        return cls(
            str(TIMESERIES_100),
            [FV.WS, FV.WD],
            var2col={FV.WS: "ws", FV.WD: "wd"},
        )
    raise AssertionError(f"Unhandled table state class {cls.__name__}")


def _multi_height_states(cls):
    if cls.__name__ == "MultiHeightNCTimeseries":
        return cls(
            data_source=str(WRF_TIMESERIES),
            time_coord="Time",
            h_coord="height",
            output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
            var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti", FV.RHO: "rho"},
        )
    if cls.__name__ == "MultiHeightNCStates":
        return cls(
            data_source=str(WRF_TIMESERIES),
            state_coord="Time",
            h_coord="height",
            output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
            var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti", FV.RHO: "rho"},
        )
    if cls.__name__ == "MultiHeightTimeseries":
        data = pd.DataFrame(
            {
                "Time": pd.date_range("2023-01-01", periods=3, freq="10min"),
                "ws-20": [6.8, 6.9, 7.0],
                "ws-90": [8.0, 8.1, 8.2],
                "ws-180": [9.0, 9.1, 9.2],
                "wd-20": [270.0, 270.0, 270.0],
                "wd-90": [270.0, 270.0, 270.0],
                "wd-180": [270.0, 270.0, 270.0],
                "ti-20": [0.07, 0.07, 0.07],
                "ti-90": [0.08, 0.08, 0.08],
                "ti-180": [0.09, 0.09, 0.09],
            }
        ).set_index("Time")
        return cls(
            data_source=data,
            output_vars=[FV.WS, FV.WD, FV.TI],
            heights=[20.0, 90.0, 180.0],
            var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
            fixed_vars={FV.RHO: 1.225},
        )
    if cls.__name__ == "MultiHeightStates":
        data = pd.DataFrame(
            {
                "state": [0, 1],
                "ws-20": [6.8, 7.0],
                "ws-90": [8.0, 8.2],
                "ws-180": [9.0, 9.2],
                "wd-20": [270.0, 270.0],
                "wd-90": [270.0, 270.0],
                "wd-180": [270.0, 270.0],
                "ti-20": [0.07, 0.07],
                "ti-90": [0.08, 0.08],
                "ti-180": [0.09, 0.09],
            }
        )
        return cls(
            data_source=data,
            output_vars=[FV.WS, FV.WD, FV.TI],
            heights=[20.0, 90.0, 180.0],
            var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
            fixed_vars={FV.RHO: 1.225},
        )
    raise AssertionError(f"Unhandled multi-height class {cls.__name__}")


def _point_cloud_states(cls):
    if cls.__name__ == "PointCloudData":
        dataset = xr.open_dataset(
            POINT_CLOUD, engine=foxes.config.nc_engine
        ).assign_coords(
            state=np.arange(100, dtype=np.int32),
            point=np.arange(30, dtype=np.int32),
        )
        return cls(
            data_source=dataset,
            output_vars=[FV.WS, FV.WD],
            states_coord="state",
            point_coord="point",
            x_ncvar="x",
            y_ncvar="y",
            h_ncvar=None,
            var2ncvar={FV.WS: "ws", FV.WD: "wd"},
        )
    if cls.__name__ == "WeibullPointCloud":
        dataset = xr.open_dataset(WEIBULL_CLOUD, engine=foxes.config.nc_engine)
        return cls(
            dataset,
            output_vars=[FV.WS, FV.WD, FV.TI],
            wd_coord="wind_direction",
            ws_coord="wind_speed",
            point_coord="wind_turbine",
            x_ncvar="x",
            y_ncvar="y",
            h_ncvar="height",
            weight_ncvar="sector_probability",
            var2ncvar={
                FV.WEIBULL_A: "weibull_a",
                FV.WEIBULL_k: "weibull_k",
                FV.TI: "turbulence_intensity",
            },
            interp_pars={"method": "nearest"},
        )
    if cls.__name__ == "TurbinePointCloud":
        dataset = (
            xr.open_dataset(POINT_CLOUD, engine=foxes.config.nc_engine)
            .isel(point=slice(0, 2))
            .rename({"point": "turbine"})
            .assign_coords(
                state=np.arange(100, dtype=np.int32),
                turbine=np.arange(2, dtype=np.int32),
            )
        )
        return cls(
            data_source=dataset,
            output_vars=[FV.WS, FV.WD],
            states_coord="state",
            turbine_coord="turbine",
            var2ncvar={FV.WS: "ws", FV.WD: "wd"},
        )
    raise AssertionError(f"Unhandled point-cloud class {cls.__name__}")


def _field_states(cls, tmp_path: Path | None = None):
    if cls.__name__ == "SingleStateField":
        dataset = xr.open_dataset(WIND_ROTATION, engine=foxes.config.nc_engine).isel(
            state=0, drop=True
        )
        return cls(
            data_source=dataset,
            output_vars=[FV.WS, FV.WD],
            var2ncvar={FV.WS: "ws", FV.WD: "wd"},
            x_coord="x",
            y_coord="y",
            h_coord="h",
            fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
            bounds_extra_space=np.inf,
            height_bounds=np.inf,
            interp_pars={"bounds_error": False},
        )
    if cls.__name__ == "FieldData":
        assert tmp_path is not None
        field_data = tmp_path / "data_0.nc"
        copyfile(WIND_ROTATION, field_data)
        return cls(
            str(tmp_path / "data_*.nc"),
            output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
            states_coord="state",
            x_coord="x",
            y_coord="y",
            h_coord="h",
            time_format=None,
            fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
            var2ncvar={FV.WS: "ws", FV.WD: "wd"},
            load_mode="preload",
            bounds_extra_space=None,
            height_bounds=None,
        )
    if cls.__name__ == "LatLonFieldData":
        lons = np.array([8.00, 8.01], dtype=float)
        lats = np.array([53.00, 53.01], dtype=float)
        ds = xr.Dataset(
            data_vars={
                "ws": (("Time", "height", "lat", "lon"), np.full((2, 3, 2, 2), 8.0)),
                "wd": (("Time", "height", "lat", "lon"), np.full((2, 3, 2, 2), 270.0)),
            },
            coords={
                "Time": pd.date_range("2023-01-01", periods=2, freq="10min"),
                "height": np.array([20.0, 90.0, 180.0]),
                "lat": lats,
                "lon": lons,
            },
        )
        return cls(
            data_source=ds,
            states_coord="Time",
            lat_coord="lat",
            lon_coord="lon",
            h_coord="height",
            output_vars=[FV.WS, FV.WD],
            fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
            var2ncvar={FV.WS: "ws", FV.WD: "wd"},
            load_mode="preload",
            utm_zone="from_grid",
        )
    if cls.__name__ == "WeibullField":
        return cls(
            str(WEIBULL_GRID),
            output_vars=[FV.WS, FV.WD, FV.TI],
            x_coord="x",
            y_coord="y",
            h_coord="height",
            wd_coord="wind_direction",
            weight_ncvar="sector_probability",
            ws_bins=np.array([0.0, 5.0, 10.0, 15.0, 20.0]),
            var2ncvar={
                FV.TI: "turbulence_intensity",
                FV.WEIBULL_A: "weibull_a",
                FV.WEIBULL_k: "weibull_k",
            },
            interp_pars={"method": "nearest"},
        )
    raise AssertionError(f"Unhandled field class {cls.__name__}")


def _other_states(cls, tmp_path):
    if cls.__name__ == "DatasetStates":
        ds = xr.Dataset(
            data_vars={
                FV.WS: ((FC.STATE,), np.array([8.0, 8.2])),
                FV.WD: ((FC.STATE,), np.array([270.0, 270.0])),
                FV.TI: ((FC.STATE,), np.array([0.08, 0.08])),
                FV.RHO: ((FC.STATE,), np.array([1.225, 1.225])),
            },
            coords={FC.STATE: np.array([0, 1], dtype=np.int32)},
        )
        return cls(ds, output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO], load_mode="preload")
    if cls.__name__ == "ScanStates":
        return cls(
            scans={FV.WS: [7.0, 8.0], FV.WD: [270.0], FV.TI: [0.08], FV.RHO: [1.225]}
        )
    if cls.__name__ == "WRGStates":
        pytest.skip(
            "No packaged WRG sample is available under foxes/data for an integrated calc_farm smoke test."
        )
    if cls.__name__ == "NEWAStates":
        foxes.config.set_utm_zone(32, "U")
        return cls(
            str(_write_newa_dataset(tmp_path)),
            output_vars=[FV.WS, FV.WD],
            fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
            load_mode="preload",
            bounds_extra_space=np.inf,
            height_bounds=None,
        )
    if cls.__name__ == "ICONStates":
        return cls(
            str(_write_icon_dataset(tmp_path)),
            output_vars=[FV.WS, FV.WD],
            fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
            load_mode="preload",
            bounds_extra_space=None,
            height_bounds=None,
            utm_zone="from_grid",
        )
    if cls.__name__ == "OnePointFlowStates":
        base = foxes.input.states.Timeseries(
            str(TIMESERIES_100),
            [FV.WS, FV.WD],
            var2col={FV.WS: "ws", FV.WD: "wd"},
            fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
        )
        return cls(ref_xy=[0.0, 0.0, 90.0], base_states=base, tl_heights=[90.0])
    if cls.__name__ == "OnePointFlowTimeseries":
        return cls(
            ref_xy=[0.0, 0.0, 90.0],
            data_source=str(TIMESERIES_100),
            output_vars=[FV.WS, FV.WD],
            var2col={FV.WS: "ws", FV.WD: "wd"},
            fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
            tl_heights=[90.0],
        )
    if cls.__name__ == "OnePointFlowMultiHeightTimeseries":
        data = pd.DataFrame(
            {
                "Time": pd.date_range("2023-01-01", periods=3, freq="10min"),
                "ws-50": [7.0, 7.1, 7.2],
                "ws-100": [8.0, 8.1, 8.2],
                "wd-50": [270.0, 270.0, 270.0],
                "wd-100": [270.0, 270.0, 270.0],
                "ti-50": [0.08, 0.08, 0.08],
                "ti-100": [0.09, 0.09, 0.09],
            }
        ).set_index("Time")
        return cls(
            ref_xy=[0.0, 0.0, 90.0],
            data_source=data,
            output_vars=[FV.WS, FV.WD, FV.TI],
            heights=[50.0, 100.0],
            var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
            fixed_vars={FV.RHO: 1.225},
            tl_heights=[90.0],
        )
    if cls.__name__ == "OnePointFlowMultiHeightNCTimeseries":
        return cls(
            ref_xy=[0.0, 0.0],
            data_source=str(WRF_TIMESERIES),
            time_coord="Time",
            h_coord="height",
            output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
            var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti", FV.RHO: "rho"},
            tl_heights=[90.0],
        )
    raise AssertionError(f"Unhandled state class {cls.__name__}")


def _run_state_smoke(cls, tmp_path):
    name = cls.__name__
    if name == "SingleStateStates":
        states = cls(ws=8.0, wd=270.0, ti=0.08, rho=1.225)
    elif name in {"StatesTable", "Timeseries", "TabStates", "WeibullSectors"}:
        states = _timeseries_states(cls)
    elif name in {
        "MultiHeightNCStates",
        "MultiHeightNCTimeseries",
        "MultiHeightStates",
        "MultiHeightTimeseries",
    }:
        states = _multi_height_states(cls)
    elif name in {"PointCloudData", "WeibullPointCloud", "TurbinePointCloud"}:
        states = _point_cloud_states(cls)
    elif name in {"SingleStateField", "FieldData", "LatLonFieldData", "WeibullField"}:
        states = _field_states(cls, tmp_path)
    else:
        states = _other_states(cls, tmp_path)

    if name in {"NEWAStates", "ICONStates", "LatLonFieldData"}:
        foxes.config.set_utm_zone(32, "U")
        xy_base = foxes.utils.from_lonlat(np.array([[8.005, 53.005]], dtype=float))[0]
        farm = _farm(
            ["NREL5MW"],
            n_turbines=2,
            H=90.0,
            xy_base=xy_base,
            xy_step=[300.0, 0.0],
        )
    elif name == "WeibullPointCloud":
        xy_base = np.array([263800.0, 6505500.0])
        farm = _farm(
            ["NREL5MW"], n_turbines=2, H=70.0, xy_base=xy_base, xy_step=[100.0, 0.0]
        )
    elif name == "WeibullField":
        farm = _farm(["NREL5MW"], n_turbines=2, H=93.0)
    elif name in {"MultiHeightNCStates", "MultiHeightNCTimeseries"}:
        farm = _farm(["NREL5MW"], n_turbines=2, H=113.0)
    else:
        farm = _farm(["NREL5MW"], n_turbines=2, H=90.0)
    rotor = "level10" if "MultiHeight" in name else "centre"
    wakes = (
        ["Bastankhah2014_linear_ka02"]
        if "MultiHeight" in name
        else ["Jensen_linear_k007"]
    )
    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model=rotor,
        wake_models=wakes,
        wake_frame="rotor_wd",
        partial_wakes=None,
        verbosity=0,
    )

    engine_type = "threads"
    with _engine(engine_type=engine_type):
        farm_results = algo.calc_farm()
        _assert_farm_results(farm_results)
        if name in {
            "PointCloudData",
            "TurbinePointCloud",
            "SingleStateField",
            "FieldData",
            "LatLonFieldData",
            "WeibullField",
        }:
            n_states = farm_results.sizes[FC.STATE]
            if name == "WeibullPointCloud":
                base_points = np.array(
                    [[263800.0, 6505500.0, 70.0], [263900.0, 6505500.0, 70.0]],
                    dtype=float,
                )
            elif name == "LatLonFieldData":
                xy = foxes.utils.from_lonlat(
                    np.array([[8.005, 53.005], [8.006, 53.005]], dtype=float)
                )
                base_points = np.column_stack([xy, np.full(2, 90.0)])
            else:
                base_points = np.array(
                    [[0.0, 0.0, 90.0], [100.0, 0.0, 90.0]], dtype=float
                )
            points = np.repeat(base_points[None, :, :], n_states, axis=0)
            point_results = algo.calc_points(farm_results, points=points)
            _assert_point_results(point_results)


def _run_vertical_profile_smoke(cls):
    name = cls.__name__
    if name == "UniformProfile":
        profile = cls(FV.WS)
    elif name == "DataProfile":
        profile = cls(
            pd.DataFrame({"z": [50.0, 90.0, 120.0], "ws": [7.0, 8.0, 8.5]}),
            variable=FV.WS,
            col_z="z",
            col_var="ws",
        )
    else:
        profile = cls()
    states = _single_state(
        profiles={FV.WS: profile},
        **{FV.H: 90.0, FV.Z0: 0.05, FV.MOL: -200.0, FV.SHEAR: 0.15},
    )
    farm = _farm(["NREL5MW"], n_turbines=1)
    algo = foxes.algorithms.Downwind(
        farm, states, wake_models=["Jensen_linear_k007"], verbosity=0
    )

    with _engine():
        farm_results = algo.calc_farm()
        _assert_farm_results(farm_results)
        points = np.array([[[0.0, 0.0, 50.0], [0.0, 0.0, 120.0]]], dtype=float)
        point_results = algo.calc_points(farm_results, points=points)
        _assert_point_results(point_results)


def _run_turbine_type_smoke(cls, tmp_path):
    name = cls.__name__
    if name == "NullType":
        ttype = cls(needs_rews2=False, needs_rews3=False)
        mbook, type_alias = _mbook_with_ttype(ttype)
        set_p = foxes.models.turbine_models.SetFarmVars()
        set_p.add_var(FV.P, np.full((1, 2), 1000.0))
        set_p.add_var(FV.CT, np.full((1, 2), 0.8))
        set_alias = _alias("set_p_ct")
        set_p.name = set_alias
        mbook.turbine_models[set_alias] = set_p
        turbine_models = [set_alias, type_alias]
        states = foxes.input.states.SingleStateStates(
            ws=8.0, wd=270.0, ti=0.08, rho=1.225
        )
    elif name == "PCtFile":
        ttype = cls(str(NREL5MW), rho=1.225)
        mbook, type_alias = _mbook_with_ttype(ttype)
        turbine_models = [type_alias]
        states = foxes.input.states.SingleStateStates(
            ws=8.0, wd=270.0, ti=0.08, rho=1.225
        )
    elif name == "CpCtFile":
        curve = pd.read_csv(NREL5MW).rename(columns={"P": "cp"})
        curve["cp"] = curve["cp"] / curve["cp"].max() * 0.45
        ttype = cls(curve, col_ws="ws", col_cp="cp", col_ct="ct", rho=1.225)
        mbook, type_alias = _mbook_with_ttype(ttype)
        turbine_models = [type_alias]
        states = foxes.input.states.SingleStateStates(
            ws=8.0, wd=270.0, ti=0.08, rho=1.225
        )
    elif name in {"PCtFromTwo", "CpCtFromTwo", "WsRho2PCtFromTwo", "WsTI2PCtFromTwo"}:
        curve = pd.read_csv(NREL5MW)
        p_file = tmp_path / "smoke_curve_P.csv"
        ct_file = tmp_path / "smoke_curve_ct.csv"
        curve[["ws", "P"]].to_csv(p_file, index=False)
        curve[["ws", "ct"]].to_csv(ct_file, index=False)
        if name == "PCtFromTwo":
            ttype = cls(str(p_file), str(ct_file), rho=1.225)
        elif name == "CpCtFromTwo":
            cp_file = tmp_path / "smoke_curve_cp.csv"
            cp = curve[["ws", "P"]].copy()
            cp["cp"] = cp["P"] / cp["P"].max() * 0.45
            cp[["ws", "cp"]].to_csv(cp_file, index=False)
            ttype = cls(str(cp_file), str(ct_file), rho=1.225)
        elif name == "WsRho2PCtFromTwo":
            ptab = curve[["ws", "P"]].rename(columns={"P": "1.225"})
            ctab = curve[["ws", "ct"]].rename(columns={"ct": "1.225"})
            ptab.to_csv(p_file, index=False)
            ctab.to_csv(ct_file, index=False)
            ttype = cls(str(p_file), str(ct_file))
        else:
            ptab = curve[["ws", "P"]].rename(columns={"P": "0.08"})
            ctab = curve[["ws", "ct"]].rename(columns={"ct": "0.08"})
            ptab.to_csv(p_file, index=False)
            ctab.to_csv(ct_file, index=False)
            ttype = cls(str(p_file), str(ct_file), rho=1.225)
        mbook, type_alias = _mbook_with_ttype(ttype)
        turbine_models = [type_alias]
        states = foxes.input.states.SingleStateStates(
            ws=8.0, wd=270.0, ti=0.08, rho=1.225
        )
    elif name == "FromLookupTable":
        curve = pd.read_csv(NREL5MW)
        ttype = cls(
            curve,
            input_vars=[FV.REWS],
            varmap={FV.REWS: "ws", FV.P: "P", FV.CT: "ct"},
            rho=1.225,
        )
        mbook, type_alias = _mbook_with_ttype(ttype)
        turbine_models = [type_alias]
        states = foxes.input.states.SingleStateStates(
            ws=8.0, wd=270.0, ti=0.08, rho=1.225
        )
    elif name == "CalculatorType":

        def _calc_type(algo, mdata, fdata, st_sel):
            p = np.zeros_like(fdata[FV.REWS])
            ct = np.zeros_like(fdata[FV.REWS])
            p[st_sel] = 1000.0
            ct[st_sel] = 0.8
            return {FV.P: p, FV.CT: ct}

        ttype = cls(
            func=_calc_type,
            out_vars=[FV.P, FV.CT],
        )
        mbook, type_alias = _mbook_with_ttype(ttype)
        turbine_models = [type_alias]
        states = foxes.input.states.SingleStateStates(
            ws=8.0, wd=270.0, ti=0.08, rho=1.225
        )
    elif name == "TBLFile":
        tbl = tmp_path / "curve.tbl"
        tbl.write_text(
            "45\n90.0 126.0 0.03 5.0\n3.0 0.80 100.0\n6.0 0.82 1200.0\n9.0 0.75 3000.0\n12.0 0.30 5000.0\n"
        )
        ttype = cls(str(tbl), rho=1.225)
        mbook, type_alias = _mbook_with_ttype(ttype)
        turbine_models = [type_alias]
        states = foxes.input.states.SingleStateStates(
            ws=8.0, wd=270.0, ti=0.08, rho=1.225
        )
    else:
        raise AssertionError(f"Unhandled turbine type {name}")

    farm = _farm(turbine_models)
    algo = foxes.algorithms.Downwind(
        farm, states, wake_models=["Jensen_linear_k007"], mbook=mbook, verbosity=0
    )
    with _engine():
        farm_results = algo.calc_farm()
    _assert_farm_results(farm_results)


def _run_turbine_model_smoke(cls, tmp_path):
    name = cls.__name__
    model_alias = _alias(name)
    model = None
    states = foxes.input.states.SingleStateStates(ws=8.0, wd=270.0, ti=0.08, rho=1.225)
    mbook, type_alias = _mbook_with_ttype()

    if name == "Calculator":
        model = cls(
            in_vars=[FV.REWS, FV.TI],
            out_vars=["SUM"],
            func=lambda rews, ti, **_: (rews + ti,),
        )
    elif name == "kTI":
        model = cls(kTI=0.2)
    elif name == "LookupTable":
        table = pd.DataFrame(
            {
                "rews": [6.0, 8.0, 10.0],
                "ti": [0.08, 0.08, 0.08],
                "boost": [0.9, 1.0, 1.1],
            }
        )
        model = cls(
            table, input_vars=[FV.REWS], output_vars=["boost"], varmap={FV.REWS: "rews"}
        )
    elif name == "PowerMask":
        set_max = foxes.models.turbine_models.SetFarmVars()
        set_max.name = _alias("set_max_p")
        set_max.add_var(FV.MAX_P, np.full((1, 2), 2000.0))
        mbook.turbine_models[set_max.name] = set_max
        model = cls()
        turbine_models = [set_max.name, type_alias, model_alias]
    elif name == "RotorCentreCalc":
        model = cls({f"{FV.WD}_HH": FV.WD, f"{FV.WS}_HH": FV.WS})
    elif name == "SectorManagement":
        rules = pd.DataFrame(
            {
                FC.TURBINE: [0, 1],
                "WD_min": [0.0, 0.0],
                "WD_max": [360.0, 360.0],
                "P": [2000.0, 2000.0],
            }
        )
        model = cls(
            rules,
            range_vars=[FV.WD],
            target_vars=[FV.P],
            col_tinds=FC.TURBINE,
        )
    elif name == "SetFarmVars":
        model = cls()
        model.add_var(FV.YAWM, np.zeros((1, 2)))
    elif name == "TableFactors":
        table = pd.DataFrame(
            {
                0.06: [1.0, 1.0],
                0.10: [1.0, 1.0],
            },
            index=[260.0, 280.0],
        )
        model = cls(table, row_var=FV.WD, col_var=FV.TI, output_vars=["factor"])
    elif name == "Thrust2Ct":
        set_t = foxes.models.turbine_models.SetFarmVars()
        set_t.name = _alias("set_t")
        set_t.add_var(FV.T, np.full((1, 2), 2.0e5))
        mbook.turbine_models[set_t.name] = set_t
        model = cls()
        turbine_models = [set_t.name, type_alias, model_alias]
    elif name == "YAW2YAWM":
        set_yaw = foxes.models.turbine_models.SetFarmVars()
        set_yaw.name = _alias("set_yaw")
        set_yaw.add_var(FV.YAW, np.full((1, 2), 5.0))
        mbook.turbine_models[set_yaw.name] = set_yaw
        model = cls()
        turbine_models = [set_yaw.name, type_alias, model_alias]
    elif name == "YAWM2YAW":
        set_yawm = foxes.models.turbine_models.SetFarmVars()
        set_yawm.name = _alias("set_yawm")
        set_yawm.add_var(FV.YAWM, np.full((1, 2), 5.0))
        mbook.turbine_models[set_yawm.name] = set_yawm
        model = cls()
        turbine_models = [set_yawm.name, type_alias, model_alias]
    elif name == "YawController":
        set_yawm = foxes.models.turbine_models.SetFarmVars(once=True)
        set_yawm.name = _alias("set_yawm")
        set_yawm.add_var(FV.YAWM, 2.0)
        mbook.turbine_models[set_yawm.name] = set_yawm
        states = foxes.input.states.Timeseries(
            str(TIMESERIES_100),
            [FV.WS, FV.WD, FV.TI, FV.RHO],
            var2col={FV.WS: "ws", FV.WD: "wd"},
            fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
        )
        model = cls(max_yaw_rate=0.5, avg_time=1)
        turbine_models = [set_yawm.name, type_alias, model_alias]
    else:
        turbine_models = [type_alias, model_alias]
        if name == "TableFactors":
            states = foxes.input.states.SingleStateStates(
                ws=8.0, wd=270.0, ti=0.08, rho=1.225
            )
        elif name == "PowerMask":
            pass
        elif name == "LookupTable":
            pass
        elif name == "SectorManagement":
            pass
        elif name == "kTI":
            pass
        elif name == "Calculator":
            pass
        elif name == "RotorCentreCalc":
            pass
        else:
            raise AssertionError(f"Unhandled turbine model {name}")

    if model is not None:
        model.name = model_alias
        mbook.turbine_models[model_alias] = model
    if name not in {"PowerMask", "Thrust2Ct", "YAW2YAWM", "YAWM2YAW", "YawController"}:
        turbine_models = [type_alias, model_alias]
    farm = _farm(turbine_models, n_turbines=2)
    if name == "YawController":
        algo = foxes.algorithms.Sequential(
            farm,
            states,
            wake_models=["Jensen_linear_k007"],
            mbook=mbook,
            verbosity=0,
        )
    else:
        algo = foxes.algorithms.Downwind(
            farm,
            states,
            wake_models=["Jensen_linear_k007"],
            mbook=mbook,
            verbosity=0,
        )
    with _engine():
        if name == "YawController":
            farm_results = next(iter(algo))
        else:
            farm_results = algo.calc_farm()
    _assert_farm_results(farm_results)


def _run_rotor_model_smoke(cls):
    name = cls.__name__
    if name == "CentreRotor":
        rotor = cls()
        states = foxes.input.states.SingleStateStates(
            ws=8.0, wd=270.0, ti=0.08, rho=1.225
        )
        mbook, type_alias = _mbook_with_ttype()
        turbine_models = [type_alias]
    elif name == "GridRotor":
        rotor = cls(4)
        states = foxes.input.states.SingleStateStates(
            ws=8.0, wd=270.0, ti=0.08, rho=1.225
        )
        mbook, type_alias = _mbook_with_ttype()
        turbine_models = [type_alias]
    elif name == "LevelRotor":
        rotor = cls(3)
        states = foxes.input.states.SingleStateStates(
            ws=8.0, wd=270.0, ti=0.08, rho=1.225
        )
        mbook, type_alias = _mbook_with_ttype()
        turbine_models = [type_alias]
    elif name == "DirectMDataInfusion":
        rotor = cls(
            svars2mdvars={FV.WS: FV.WS, FV.WD: FV.WD, FV.TI: FV.TI, FV.RHO: FV.RHO}
        )
        sdata = pd.DataFrame(
            {
                "state": [0, 1],
                "ws": [8.0, 8.2],
                "wd": [270.0, 270.0],
                "ti": [0.08, 0.08],
                "rho": [1.225, 1.225],
                "weight": [0.5, 0.5],
            }
        )
        states = foxes.input.states.StatesTable(
            sdata,
            output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO, FV.WEIGHT],
            var2col={
                FV.WS: "ws",
                FV.WD: "wd",
                FV.TI: "ti",
                FV.RHO: "rho",
                FV.WEIGHT: "weight",
            },
        )
        ttype = foxes.models.turbine_types.NullType(
            needs_rews2=False, needs_rews3=False
        )
        mbook, type_alias = _mbook_with_ttype(ttype)
        setvals = foxes.models.turbine_models.SetFarmVars()
        setvals.name = _alias("direct_infusion_setvals")
        setvals.add_var(FV.P, np.full((2, 2), 1500.0))
        setvals.add_var(FV.CT, np.full((2, 2), 0.8))
        mbook.turbine_models[setvals.name] = setvals
        turbine_models = [setvals.name, type_alias]
    else:
        raise AssertionError(f"Unhandled rotor model {name}")

    alias = _alias(name)
    rotor.name = alias
    mbook.rotor_models[alias] = rotor
    farm = _farm(turbine_models)
    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model=alias,
        wake_models=["Bastankhah2014_linear_k004"],
        mbook=mbook,
        verbosity=0,
    )
    with _engine():
        farm_results = algo.calc_farm()
    _assert_farm_results(farm_results)


def _run_partial_wakes_smoke(cls):
    name = cls.__name__
    if name == "PartialAxiwake":
        model = cls(6)
    elif name == "PartialGaussianLookup":
        r_axis, s_axis = foxes.utils.create_lookup_axes(
            r_over_sigma_max=8.0,
            n_r=17,
            sigma_over_d_min=0.03,
        )
        model = cls(
            foxes.utils.build_lookup_dataset(
                r_over_sigma=r_axis,
                sigma_over_d=s_axis,
                n_rho=96,
                version_tag="smoke-v1",
            )
        )
    elif name == "PartialSegregated":
        model = cls(foxes.models.rotor_models.CentreRotor())
    elif name == "PartialGrid":
        model = cls(4)
    else:
        model = cls()
    mbook, type_alias = _mbook_with_ttype()
    alias = _alias(name)
    model.name = alias
    mbook.partial_wakes[alias] = model
    farm = _farm([type_alias])
    states = foxes.input.states.SingleStateStates(ws=8.0, wd=270.0, ti=0.08, rho=1.225)
    wake_alias = (
        "Jensen_linear_k007"
        if name == "PartialTopHat"
        else "Bastankhah2014_linear_k004"
    )
    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model="centre",
        wake_models=[wake_alias],
        partial_wakes={wake_alias: alias},
        mbook=mbook,
        verbosity=0,
    )
    with _engine():
        farm_results = algo.calc_farm()
    _assert_farm_results(farm_results)


def _run_point_model_smoke(cls):
    name = cls.__name__
    if name == "SetUniformData":
        data = pd.DataFrame({"ws": [8.0], "wd": [270.0]})
        model = cls(
            data_source=data,
            output_vars=[FV.WS, FV.WD],
            var2col={FV.WS: "ws", FV.WD: "wd"},
        )
        states = foxes.input.states.SingleStateStates(
            ws=8.0, wd=270.0, ti=0.08, rho=1.225
        )
    elif name == "TKE2TI":
        model = cls()
        states = foxes.input.states.SingleStateStates(
            ws=8.0,
            wd=270.0,
            rho=1.225,
            profiles={FV.TKE: foxes.models.vertical_profiles.UniformProfile(FV.TKE)},
            TKE=0.5,
        )
    elif name == "Ustar2TI":
        model = cls()
        states = foxes.input.states.SingleStateStates(
            ws=8.0,
            wd=270.0,
            rho=1.225,
            profiles={
                FV.USTAR: foxes.models.vertical_profiles.UniformProfile(FV.USTAR)
            },
            USTAR=0.4,
        )
    elif name == "WakeDeltas":
        model = cls(vars=[FV.WS])
        states = foxes.input.states.SingleStateStates(
            ws=8.0, wd=270.0, ti=0.08, rho=1.225
        )
    else:
        raise AssertionError(f"Unhandled point model {name}")

    mbook, type_alias = _mbook_with_ttype()
    alias = _alias(name)
    model.name = alias
    mbook.point_models[alias] = model
    farm = _farm([type_alias], n_turbines=1)
    algo = foxes.algorithms.Downwind(
        farm, states, wake_models=["Jensen_linear_k007"], mbook=mbook, verbosity=0
    )
    with _engine():
        farm_results = algo.calc_farm()
        points = np.array([[[0.0, 0.0, 90.0], [100.0, 0.0, 90.0]]], dtype=float)
        point_results = algo.calc_points(
            farm_results, points=points, point_models=[alias]
        )
    _assert_point_results(point_results)
    if name == "WakeDeltas":
        assert "DELTA_WS" in point_results


def _run_wake_superposition_smoke(cls):
    name = cls.__name__
    if name in {"WSPow", "WSPowLocal", "TIPow"}:
        model = cls(pow=3)
    else:
        model = cls()
    mbook, type_alias = _mbook_with_ttype()
    superp_alias = _alias(name)
    model.name = superp_alias
    mbook.wake_superpositions[superp_alias] = model

    if name.startswith("TI"):
        wake = foxes.models.wake_models.ti.IECTIWake(superposition=superp_alias)
    elif name == "WindVectorLinear":
        wake = foxes.models.wake_models.induction.RankineHalfBody(
            superposition=superp_alias
        )
    else:
        wake = foxes.models.wake_models.wind.JensenWake(
            superposition=superp_alias,
            k=0.04,
        )
    wake_alias = _alias(f"{name}_wake")
    wake.name = wake_alias
    mbook.wake_models[wake_alias] = wake

    farm = _farm([type_alias])
    states = foxes.input.states.SingleStateStates(ws=8.0, wd=270.0, ti=0.08, rho=1.225)
    algo = foxes.algorithms.Downwind(
        farm, states, wake_models=[wake_alias], mbook=mbook, verbosity=0
    )
    with _engine():
        farm_results = algo.calc_farm()
    _assert_farm_results(farm_results)


def _run_axial_induction_smoke(cls):
    model = cls()
    mbook, type_alias = _mbook_with_ttype()
    ind_alias = _alias(cls.__name__)
    model.name = ind_alias
    mbook.axial_induction[ind_alias] = model
    super_alias = _alias("ws_linear")
    if super_alias not in mbook.wake_superpositions:
        mbook.wake_superpositions[super_alias] = (
            foxes.models.wake_superpositions.WSLinear()
        )
    wake = foxes.models.wake_models.wind.JensenWake(
        superposition=super_alias,
        induction=ind_alias,
        k=0.04,
    )
    wake_alias = _alias(f"{cls.__name__}_wake")
    wake.name = wake_alias
    mbook.wake_models[wake_alias] = wake
    farm = _farm([type_alias])
    states = foxes.input.states.SingleStateStates(ws=8.0, wd=270.0, ti=0.08, rho=1.225)
    algo = foxes.algorithms.Downwind(
        farm, states, wake_models=[wake_alias], mbook=mbook, verbosity=0
    )
    with _engine():
        farm_results = algo.calc_farm()
    _assert_farm_results(farm_results)


def _run_ground_model_smoke(cls):
    if cls.__name__ == "WakeMirror":
        model = cls([0.0])
    else:
        model = cls()
    mbook, type_alias = _mbook_with_ttype()
    alias = _alias(cls.__name__)
    model.name = alias
    mbook.ground_models[alias] = model
    farm = _farm([type_alias])
    states = foxes.input.states.SingleStateStates(ws=8.0, wd=270.0, ti=0.08, rho=1.225)
    algo = foxes.algorithms.Downwind(
        farm,
        states,
        wake_models=["Jensen_linear_k007"],
        ground_models={"Jensen_linear_k007": alias},
        mbook=mbook,
        verbosity=0,
    )
    with _engine():
        farm_results = algo.calc_farm()
    _assert_farm_results(farm_results)


def _run_wake_deflection_smoke(cls):
    model = cls()
    mbook, type_alias = _mbook_with_ttype()
    set_yaw = foxes.models.turbine_models.SetFarmVars()
    set_yaw.name = _alias("set_yaw")
    set_yaw.add_var(FV.YAWM, np.full((1, 2), 10.0))
    mbook.turbine_models[set_yaw.name] = set_yaw
    yawm2yaw = foxes.models.turbine_models.YAWM2YAW()
    yawm2yaw.name = _alias("yawm2yaw")
    mbook.turbine_models[yawm2yaw.name] = yawm2yaw
    alias = _alias(cls.__name__)
    model.name = alias
    mbook.wake_deflections[alias] = model
    wake = foxes.models.wake_models.wind.JensenWake(superposition="vector", k=0.04)
    wake_alias = _alias("jimenez_wake")
    wake.name = wake_alias
    mbook.wake_models[wake_alias] = wake
    farm = _farm([set_yaw.name, yawm2yaw.name, type_alias])
    states = foxes.input.states.SingleStateStates(ws=8.0, wd=270.0, ti=0.08, rho=1.225)
    algo = foxes.algorithms.Downwind(
        farm,
        states,
        wake_models=[wake_alias],
        partial_wakes="centre",
        wake_deflection=alias,
        mbook=mbook,
        verbosity=0,
    )
    with _engine():
        farm_results = algo.calc_farm()
    _assert_farm_results(farm_results)


def _run_wake_frame_smoke(cls):
    name = cls.__name__
    if name == "RotorWD":
        frame = cls()
        states = foxes.input.states.SingleStateStates(
            ws=8.0, wd=270.0, ti=0.08, rho=1.225
        )
        algo_cls = foxes.algorithms.Downwind
        extra = {}
    elif name == "FarmOrder":
        frame = cls()
        states = foxes.input.states.SingleStateStates(
            ws=8.0, wd=270.0, ti=0.08, rho=1.225
        )
        algo_cls = foxes.algorithms.Downwind
        extra = {}
    elif name == "Streamlines2D":
        frame = cls(step=50.0)
        states = _field_states(foxes.input.states.SingleStateField)
        algo_cls = foxes.algorithms.Downwind
        extra = {}
    elif name == "Timelines":
        frame = cls(dt_min=10.0)
        states = foxes.input.states.Timeseries(
            str(TIMESERIES_100),
            [FV.WS, FV.WD, FV.TI, FV.RHO],
            var2col={FV.WS: "ws", FV.WD: "wd"},
            fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
        )
        algo_cls = foxes.algorithms.Iterative
        extra = {"max_wake_length_km": 5.0}
    elif name == "DynamicWakes":
        frame = cls(dt_min=10.0)
        states = foxes.input.states.Timeseries(
            str(TIMESERIES_100),
            [FV.WS, FV.WD, FV.TI, FV.RHO],
            var2col={FV.WS: "ws", FV.WD: "wd"},
            fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
        )
        algo_cls = foxes.algorithms.Iterative
        extra = {"max_wake_length_km": 5.0}
    elif name == "SeqDynamicWakes":
        frame = cls(dt_min=10.0)
        states = foxes.input.states.Timeseries(
            str(TIMESERIES_100),
            [FV.WS, FV.WD, FV.TI, FV.RHO],
            var2col={FV.WS: "ws", FV.WD: "wd"},
            fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
        )
        algo_cls = foxes.algorithms.Sequential
        extra = {"max_wake_length_km": 5.0}
    else:
        raise AssertionError(f"Unhandled wake frame {name}")

    mbook, type_alias = _mbook_with_ttype()
    alias = _alias(name)
    frame.name = alias
    mbook.wake_frames[alias] = frame
    farm = _farm([type_alias])
    if name == "Streamlines2D":
        extra = {**extra, "max_wake_length_km": 5.0}
    algo = algo_cls(
        farm,
        states,
        wake_models=["Jensen_linear_k007"],
        wake_frame=alias,
        mbook=mbook,
        verbosity=0,
        **extra,
    )
    with _engine():
        if name == "SeqDynamicWakes":
            farm_results = next(iter(algo))
        else:
            farm_results = algo.calc_farm()
    _assert_farm_results(farm_results)


def _run_wake_model_smoke(cls):
    name = cls.__name__
    mbook, type_alias = _mbook_with_ttype()
    if name in {"CrespoHernandezTIWake", "IECTIWake"}:
        if name == "IECTIWake":
            model = cls(superposition="ti_quadratic")
        else:
            model = cls(superposition="ti_quadratic", k=0.04)
    elif name == "TurbOParkWakeIX":
        model = cls(superposition="ws_linear", dx=40.0, k=0.04)
    elif name == "RankineHalfBody":
        model = cls(superposition="vector")
    elif name in {
        "JensenWake",
        "JensenTurbOParkWake",
        "Bastankhah2014",
        "Bastankhah2016",
        "TurbOParkWake",
    }:
        model = cls(superposition="ws_linear", k=0.04)
    else:
        model = cls(superposition="ws_linear")
    alias = _alias(name)
    model.name = alias
    mbook.wake_models[alias] = model
    wake_models = [alias]
    if name == "TurbOParkWakeIX":
        ti_wake = foxes.models.wake_models.ti.IECTIWake(superposition="ti_quadratic")
        ti_alias = _alias("turboparkix_ti")
        ti_wake.name = ti_alias
        mbook.wake_models[ti_alias] = ti_wake
        wake_models.append(ti_alias)
    farm = _farm([type_alias])
    states = foxes.input.states.SingleStateStates(ws=8.0, wd=270.0, ti=0.08, rho=1.225)
    extra = {}
    if name == "TurbOParkWakeIX":
        extra["max_wake_length_km"] = 20.0
    algo = foxes.algorithms.Downwind(
        farm,
        states,
        wake_models=wake_models,
        mbook=mbook,
        verbosity=0,
        **extra,
    )
    with _engine():
        farm_results = algo.calc_farm()
    _assert_farm_results(farm_results)


def _run_farm_controller_smoke(cls):
    name = cls.__name__
    if name == "BasicFarmController":
        controller = cls()
        states = foxes.input.states.SingleStateStates(
            ws=8.0, wd=270.0, ti=0.08, rho=1.225
        )
    elif name == "OpFlagController":
        controller = cls(data_source=np.array([[True, False]], dtype=bool))
        states = foxes.input.states.SingleStateStates(
            ws=8.0, wd=270.0, ti=0.08, rho=1.225
        )
    else:
        raise AssertionError(f"Unhandled farm controller {name}")

    mbook, type_alias = _mbook_with_ttype()
    alias = _alias(name)
    controller.name = alias
    mbook.farm_controllers[alias] = controller
    farm = _farm([type_alias])
    algo = foxes.algorithms.Downwind(
        farm,
        states,
        wake_models=["Jensen_linear_k007"],
        farm_controller=alias,
        mbook=mbook,
        verbosity=0,
    )
    with _engine():
        farm_results = algo.calc_farm()
    _assert_farm_results(farm_results)


def run_model_smoke(model_path: str, tmp_path: Path):
    if model_path in UNSUPPORTED:
        pytest.skip(UNSUPPORTED[model_path])

    cls = _load_class(model_path)
    module_name = cls.__module__

    if module_name.startswith("foxes.input.states."):
        _run_state_smoke(cls, tmp_path)
    elif module_name.startswith("foxes.models.vertical_profiles."):
        _run_vertical_profile_smoke(cls)
    elif module_name.startswith("foxes.models.turbine_types."):
        _run_turbine_type_smoke(cls, tmp_path)
    elif module_name.startswith("foxes.models.turbine_models."):
        _run_turbine_model_smoke(cls, tmp_path)
    elif module_name.startswith("foxes.models.rotor_models."):
        _run_rotor_model_smoke(cls)
    elif module_name.startswith("foxes.models.partial_wakes."):
        _run_partial_wakes_smoke(cls)
    elif module_name.startswith("foxes.models.point_models."):
        _run_point_model_smoke(cls)
    elif module_name.startswith("foxes.models.wake_superpositions."):
        _run_wake_superposition_smoke(cls)
    elif module_name.startswith("foxes.models.axial_induction."):
        _run_axial_induction_smoke(cls)
    elif module_name.startswith("foxes.models.ground_models."):
        _run_ground_model_smoke(cls)
    elif module_name.startswith("foxes.models.wake_deflections."):
        _run_wake_deflection_smoke(cls)
    elif module_name.startswith("foxes.models.wake_frames."):
        _run_wake_frame_smoke(cls)
    elif module_name.startswith("foxes.models.wake_models."):
        _run_wake_model_smoke(cls)
    elif module_name.startswith("foxes.models.farm_controllers."):
        _run_farm_controller_smoke(cls)
    elif module_name.startswith("foxes.models.farm_models."):
        pytest.skip(
            UNSUPPORTED.get(
                model_path,
                "No calc_farm/calc_points integration path is available for this public helper model.",
            )
        )
    else:
        raise AssertionError(f"Unhandled model path {model_path}")
