from types import SimpleNamespace

import numpy as np
import pandas as pd

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import FData
from foxes.models.turbine_models.sector_management import SectorManagement


def test_sector_management_sets_target_in_defined_ranges():
    df = pd.DataFrame(
        {
            "WD_min": [350.0, 90.0],
            "WD_max": [10.0, 210.0],
            FV.MAX_P: [1000.0, 2000.0],
        },
        index=pd.Index([0, 1], name=FC.TURBINE),
    )

    model = SectorManagement(
        data_source=df,
        range_vars=[FV.WD],
        target_vars=[FV.MAX_P],
        col_tinds=FC.TURBINE,
    )
    algo = SimpleNamespace(store_model_data=lambda *args, **kwargs: None)
    model.initialize(algo=algo, verbosity=0)

    fdata = FData(
        data={
            FV.WD: np.array([[355.0, 100.0], [5.0, 250.0]], dtype=float),
            FV.MAX_P: np.full((2, 2), np.nan, dtype=float),
        },
        dims={
            FV.WD: (FC.STATE, FC.TURBINE),
            FV.MAX_P: (FC.STATE, FC.TURBINE),
        },
    )
    st_sel = np.array([[True, True], [True, True]])

    out = model.calculate(algo=None, mdata={}, fdata=fdata, st_sel=st_sel)

    assert np.isclose(out[FV.MAX_P][0, 0], 1000.0)
    assert np.isclose(out[FV.MAX_P][1, 0], 1000.0)
    assert np.isclose(out[FV.MAX_P][0, 1], 2000.0)
    assert np.isnan(out[FV.MAX_P][1, 1])


def main():
    test_sector_management_sets_target_in_defined_ranges()


if __name__ == "__main__":
    main()
