import numpy as np
import xarray as xr

from foxes.input.states.point_cloud_data import PointCloudData
import foxes.constants as FC
import foxes.variables as FV


class _AlgoMock:
    pass


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
