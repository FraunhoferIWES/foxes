import numpy as np
import xarray as xr

import foxes.constants as FC
import foxes.variables as FV
from foxes.algorithms.downwind.models.reorder_farm_output import ReorderFarmOutput
from foxes.core import FData, MData
from foxes.input.states import DatasetStates


def test_reorder_farm_output_derives_inverse_order():
    order = np.array([[2, 0, 1], [1, 2, 0]])
    fdata = FData(
        data={
            FV.ORDER: order,
            FV.WS: np.array([[30.0, 10.0, 20.0], [50.0, 60.0, 40.0]]),
        },
        dims={
            FV.ORDER: (FC.STATE, FC.TURBINE),
            FV.WS: (FC.STATE, FC.TURBINE),
        },
    )
    model = ReorderFarmOutput(outputs=[FV.WS])

    results = model.calculate(algo=None, mdata={}, fdata=fdata)

    np.testing.assert_allclose(
        results[FV.WS],
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
    )


def test_turbine_only_data_follows_each_states_downwind_order():
    states = DatasetStates(
        data_source=xr.Dataset(),
        output_vars=[FV.WS],
        load_mode="preload",
    )
    states._cmap = {FC.STATE: FC.STATE}
    states.META = states.var("meta")
    data_key = states.var("data0")
    vars_key = states.var("vars0")
    mdata = MData(
        data={
            FC.STATE: np.arange(2),
            vars_key: np.array([FV.WS]),
            data_key: np.array([[8.0], [9.0], [10.0]]),
        },
        dims={
            FC.STATE: (FC.STATE,),
            vars_key: (vars_key,),
            data_key: (FC.TURBINE, vars_key),
        },
        extra_data={states.META: {"data_keys": [data_key]}},
    )
    fdata = FData(
        data={FV.ORDER: np.array([[2, 0, 1], [1, 2, 0]])},
        dims={FV.ORDER: (FC.STATE, FC.TURBINE)},
    )

    data, _ = states._get_calc_data(mdata, fdata)

    variables, values = data[(FC.STATE, FC.TURBINE, vars_key)]
    assert variables == [FV.WS]
    np.testing.assert_allclose(
        values[..., 0],
        [[10.0, 8.0, 9.0], [9.0, 10.0, 8.0]],
    )
