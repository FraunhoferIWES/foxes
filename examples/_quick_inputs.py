"""Small, local inputs for the example runner's explicit quick mode."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

import foxes
from foxes.data import FARM, STATES


def prepare_inputs(directory: Path, example: str) -> None:
    """Write bounded public/synthetic fixtures inside an isolated example copy."""
    data = foxes.StaticData()
    times = pd.date_range("2020-01-01", periods=3, freq="min")
    frame = pd.DataFrame(
        {
            "ws": [7.0, 8.0, 9.0],
            "wd": [270.0, 275.0, 280.0],
            "ti": [0.07] * 3,
            "rho": [1.225] * 3,
            "mol": [1000.0] * 3,
        },
        index=times,
    )
    for lower, upper in (("ws", "WS"), ("wd", "WD"), ("ti", "TI"), ("rho", "RHO")):
        frame[upper] = frame[lower]
    for name in (
        "timeseries_3000.csv.gz",
        "timeseries_8000.csv.gz",
        "timeseries_100.csv.gz",
        "abl_states_6000.csv.gz",
        "10s_TEST.csv",
    ):
        frame.to_csv(directory / name)
    if example == "power_mask":
        pd.DataFrame({"WS": [7.0, 8.0, 9.0]}).to_csv(directory / "states.csv")
        pd.DataFrame({f"Pmax_{i}": [3000.0] * 3 for i in range(5)}).to_csv(
            directory / "power_mask.csv", index=False
        )

    # Keep packaged layouts recognizable but bound the state/turbine product.
    layout_path = data.get_file_path(FARM, "test_farm_67.csv")
    layout = pd.read_csv(layout_path).iloc[:2]
    layout.to_csv(directory / "test_farm_67.csv", index=False)
    for name in ("wind_rose_bremen.csv", "weibull_sectors_12.csv"):
        pd.read_csv(data.get_file_path(STATES, name)).iloc[:2].to_csv(
            directory / name, index=False
        )
    for name in (
        "WRF-Timeseries-3000.nc",
        "point_cloud_100.nc",
        "weibull_grid.nc",
        "weibull_cloud_4.nc",
    ):
        with xr.open_dataset(data.get_file_path(STATES, name)) as dataset:
            selections = {
                dim: slice(0, 2)
                for dim in dataset.dims
                if dim.lower() in {"time", "state", "wind_direction"}
            }
            dataset.isel(selections).load().to_netcdf(directory / name)

    # Cartesian fields span the actual tutorial farm coordinates.
    coordinates = {
        "state": [0, 1, 2],
        "h": [0.0, 300.0],
        "y": [-10000.0, 1000.0, 10000.0],
        "x": [-10000.0, 1000.0, 10000.0],
    }
    shape = tuple(len(v) for v in coordinates.values())
    values = np.broadcast_to(np.array([7.0, 8.0, 9.0])[:, None, None, None], shape)
    field = xr.Dataset(
        {
            name: (tuple(coordinates), np.full(shape, value))
            for name, value in {"wd": 270.0, "TI": 0.08, "RHO": 1.225}.items()
        },
        coords=coordinates,
    )
    field["ws"] = (tuple(coordinates), values)
    field.to_netcdf(directory / "wind_rotation.nc")
    if example == "field_data_nc":
        field = field.rename(
            {
                "state": "Time",
                "h": "height",
                "x": "UTMX",
                "y": "UTMY",
                "ws": "WS",
                "wd": "WD",
            }
        )
        field = field.assign_coords(Time=times)
        field.to_netcdf(directory / "field.nc")
    if example == "icon_dream":
        _icon_inputs(directory, times)
    if example == "windio":
        _windio_inputs(directory)
    if (directory / "inputs.yaml").is_file():
        _bound_yaml(directory)


def _bound_yaml(directory: Path) -> None:
    """Bound YAML geometry/bin counts without replacing its state-provider type."""
    path = directory / "inputs.yaml"
    configuration = yaml.safe_load(path.read_text())
    states = configuration["states"]
    if "ws_bins" in states:
        states["ws_bins"] = [0, 8, 30]
    layouts = configuration["wind_farm"]["layouts"]
    for layout in layouts:
        if "steps" in layout:
            layout["steps"] = [1, 2]
    configuration["engine"] = {
        "engine_type": "numpy",
        "chunk_size_states": 2,
        "chunk_size_points": 500,
        "verbosity": 0,
    }
    path.write_text(yaml.safe_dump(configuration, sort_keys=False))


def _icon_inputs(directory: Path, times: pd.DatetimeIndex) -> None:
    """Create an ICON grid and EWW-style farm/curve using synthetic values."""
    pd.DataFrame(
        {
            "wind_farm": ["Synthetic"] * 2,
            "latitude": [54.5, 54.5],
            "longitude": [12.5, 12.51],
            "commissioning_date": ["2020-01-01"] * 2,
            "turbine_type": ["Synthetic"] * 2,
            "rated_power": [5.0] * 2,
            "rotor_diameter": [126.0] * 2,
            "hub_height": [90.0] * 2,
        }
    ).to_csv(directory / "farms.csv")
    curves = directory / "curves"
    curves.mkdir()
    pd.DataFrame(
        {
            "ws": [0, 3, 8, 12, 25, 30],
            "power": [0, 0, 2000, 5000, 5000, 0],
            "ct": [0, 0.8, 0.8, 0.5, 0.1, 0],
        }
    ).to_csv(curves / "Synthetic.csv", index=False)
    shape = (3, 4, 2, 2)
    icon = xr.Dataset(
        {
            name: (("time", "height", "lat", "lon"), np.full(shape, value))
            for name, value in {
                "U": 8.0,
                "V": 0.0,
                "T": 288.15,
                "p": 101325.0,
                "RHO": 1.225,
            }.items()
        },
        coords={
            "time": times,
            "height": [71, 72, 73, 74],
            "height_2": [71, 72, 73, 74],
            "lat": [54.4, 54.6],
            "lon": [12.4, 12.6],
        },
    )
    icon["TKE"] = (("time", "height_2", "lat", "lon"), np.full(shape, 0.3))
    icon.to_netcdf(directory / "icon.nc")


def _windio_inputs(directory: Path) -> None:
    """Inline public WindIO includes so quick mode needs no optional YAML loader."""

    class Loader(yaml.SafeLoader):
        pass

    def include(loader, node):
        path = directory / loader.construct_scalar(node)
        if path.suffix == ".nc":
            with xr.open_dataset(path) as dataset:
                dataset = dataset.isel(time=slice(0, 3))
                return {
                    **{
                        name: data.values.tolist()
                        for name, data in dataset.coords.items()
                    },
                    **{
                        name: {"dims": list(data.dims), "data": data.values.tolist()}
                        for name, data in dataset.data_vars.items()
                    },
                }
        return yaml.safe_load(path.read_text())

    Loader.add_constructor("!include", include)
    configuration = yaml.load(
        (directory / "windio_inputs.yaml").read_text(), Loader=Loader
    )
    specification = configuration["attributes"]["model_outputs_specification"]
    specification["run_configuration"]["times_run"] = {"all_occurences": True}
    specification["flow_field"]["report"] = False
    path = directory / "quick_windio.yaml"
    path.write_text(yaml.safe_dump(configuration, sort_keys=False))
