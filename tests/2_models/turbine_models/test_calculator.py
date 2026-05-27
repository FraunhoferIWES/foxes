import numpy as np
from types import SimpleNamespace

import foxes.constants as FC
from foxes.core import FData
from foxes.models.turbine_models.calculator import Calculator


def test_calculator_calculate():
    model = Calculator(
        in_vars=["in0", "in1"],
        out_vars=["out0", "out1"],
        func=lambda a, b, **_: (a + b, a - b),
    )
    algo = SimpleNamespace(store_model_data=lambda *args, **kwargs: None)
    model.initialize(algo=algo, verbosity=0)

    fdata = FData(
        data={
            "in0": np.array([[2.0, 3.0], [4.0, 5.0]]),
            "in1": np.array([[1.0, 1.5], [2.0, 2.5]]),
        },
        dims={
            "in0": (FC.STATE, FC.TURBINE),
            "in1": (FC.STATE, FC.TURBINE),
        },
    )

    out = model.calculate(algo=None, mdata={}, fdata=fdata, st_sel=np.s_[:, :])

    assert set(out) == {"out0", "out1"}
    assert np.allclose(out["out0"], np.array([[3.0, 4.5], [6.0, 7.5]]))
    assert np.allclose(out["out1"], np.array([[1.0, 1.5], [2.0, 2.5]]))


def main():
    test_calculator_calculate()


if __name__ == "__main__":
    main()
