import pandas as pd
import xarray as xr

import foxes.constants as FC
import foxes.variables as FV
from foxes.input.states import StatesTable


def test_states_table_weight_dim_is_state_tuple():
    states = StatesTable(
        data_source=pd.DataFrame(
            {
                "ws": [8.0, 9.0],
                "wd": [270.0, 280.0],
                "weight": [0.4, 0.6],
            }
        ),
        output_vars=[FV.WS, FV.WD],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.WEIGHT: "weight"},
    )

    loaded_data = states.initialize(algo=object(), verbosity=0)

    weight_var = states.var(FV.WEIGHT)
    assert weight_var in loaded_data["data_vars"]

    dims, weights = loaded_data["data_vars"][weight_var]
    assert dims == (FC.STATE,)
    assert weights.shape == (2,)

    ds = xr.Dataset(coords=loaded_data["coords"], data_vars=loaded_data["data_vars"])
    assert ds[weight_var].dims == (FC.STATE,)
