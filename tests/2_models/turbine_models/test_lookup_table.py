import numpy as np
import pandas as pd
from types import SimpleNamespace

import foxes.constants as FC
from foxes.core import FData
from foxes.models.turbine_models.lookup_table import LookupTable


def test_lookup_table_interpolation():
    in_var = "in_x"
    out_var = "out_y"
    sdata = pd.DataFrame({in_var: [0.0, 10.0], out_var: [1.0, 3.0]})
    model = LookupTable(
        data_source=sdata,
        input_vars=[in_var],
        output_vars=[out_var],
        interpn_args={"bounds_error": False, "fill_value": None},
    )
    algo = SimpleNamespace(store_model_data=lambda *args, **kwargs: None)
    model.initialize(algo=algo, verbosity=0)

    st_sel = np.array([[True, False], [True, True]])
    fdata = FData(
        data={
            in_var: np.array([[2.5, 0.0], [5.0, 7.5]], dtype=float),
            out_var: np.zeros((2, 2), dtype=float),
        },
        dims={
            in_var: (FC.STATE, FC.TURBINE),
            out_var: (FC.STATE, FC.TURBINE),
        },
    )

    out = model.calculate(algo=algo, mdata={}, fdata=fdata, st_sel=st_sel)

    expected = np.zeros((2, 2), dtype=float)
    expected[st_sel] = np.array([1.5, 2.0, 2.5])
    assert np.allclose(out[out_var], expected)


def main():
    test_lookup_table_interpolation()


if __name__ == "__main__":
    main()
