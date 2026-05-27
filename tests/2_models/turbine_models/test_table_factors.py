from types import SimpleNamespace

import numpy as np
import pandas as pd

import foxes.constants as FC
from foxes.core import FData
from foxes.models.turbine_models.table_factors import TableFactors


def test_table_factors_multiplies_outputs():
    df = pd.DataFrame(
        data=[[1.0, 2.0], [3.0, 4.0]],
        index=np.array([0.0, 10.0]),
        columns=np.array([0.0, 20.0]),
    )
    model = TableFactors(
        data_source=df,
        row_var="r",
        col_var="c",
        output_vars=["out"],
        bounds_error=False,
        fill_value=None,
    )
    algo = SimpleNamespace(store_model_data=lambda *args, **kwargs: None)
    model.initialize(algo=algo, verbosity=0)

    st_sel = np.array([[True, False], [True, True]])
    fdata = FData(
        data={
            "r": np.array([[0.0, 0.0], [10.0, 0.0]], dtype=float),
            "c": np.array([[0.0, 0.0], [20.0, 20.0]], dtype=float),
            "out": np.ones((2, 2), dtype=float),
        },
        dims={
            "r": (FC.STATE, FC.TURBINE),
            "c": (FC.STATE, FC.TURBINE),
            "out": (FC.STATE, FC.TURBINE),
        },
    )

    out = model.calculate(algo=None, mdata={}, fdata=fdata, st_sel=st_sel)

    expected = np.ones((2, 2), dtype=float)
    expected[0, 0] = 1.0
    expected[1, 0] = 4.0
    expected[1, 1] = 2.0
    assert np.allclose(out["out"], expected)


def main():
    test_table_factors_multiplies_outputs()


if __name__ == "__main__":
    main()
