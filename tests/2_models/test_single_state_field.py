from types import SimpleNamespace
import warnings

import numpy as np
import xarray as xr

from foxes.core import TData
from foxes.input.states.single_state_field import SingleStateField
import foxes.variables as FV


def test_single_state_field_skips_nan_points_without_all_nan_warning():
    data_source = xr.Dataset(
        data_vars={
            "ws": (("x", "y", "height"), np.array([[[8.0]], [[9.0]]])),
            "wd": (("x", "y", "height"), np.array([[[270.0]], [[271.0]]])),
        },
        coords={
            "x": np.array([0.0, 100.0]),
            "y": np.array([0.0]),
            "height": np.array([90.0]),
        },
    )

    states = SingleStateField(
        data_source=data_source,
        output_vars=[FV.WS, FV.WD],
        var2ncvar={FV.WS: "ws", FV.WD: "wd"},
        fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
        bounds_extra_space=np.inf,
        height_bounds=np.inf,
    )

    loaded = {"coords": {}, "data_vars": {}, "extra_data": {}}
    states.load_data(algo=None, loaded_data=loaded, verbosity=0)
    mdata = SimpleNamespace(extra_data=loaded["extra_data"])

    points = np.array([[[np.nan, 0.0, 90.0], [0.0, 0.0, 90.0]]], dtype=float)
    tdata = TData.from_points(points=points)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        states.calculate(algo=None, mdata=mdata, fdata=None, tdata=tdata)

    msgs = [str(w.message) for w in caught]
    assert not any("All-NaN slice encountered" in m for m in msgs)
    assert np.isnan(tdata[FV.WS][0, 0, 0])
    assert np.isfinite(tdata[FV.WS][0, 1, 0])
