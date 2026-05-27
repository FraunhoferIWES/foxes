from types import SimpleNamespace

import numpy as np

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import FData
from foxes.models.turbine_models.rotor_centre_calc import RotorCentreCalc


class DummyWakeCalc:
    initialized = False
    running = False

    def initialize(self, algo, verbosity=0, force=False):
        self.initialized = True

    def calculate(self, algo, mdata, fdata, tdata):
        vals = np.array([[[10.0], [20.0]], [[30.0], [40.0]]], dtype=float)
        return {"rotor_v": vals}


def test_rotor_centre_calc_extracts_selected_values():
    model = RotorCentreCalc(calc_vars={"out_v": "rotor_v"})
    algo = SimpleNamespace(
        store_model_data=lambda *args, **kwargs: None,
        get_model=lambda _name: (lambda: DummyWakeCalc()),
        states=SimpleNamespace(calculate=lambda *args, **kwargs: {}),
    )
    model.initialize(algo=algo, verbosity=0)
    st_sel = np.array([[True, False], [False, True]])

    fdata = FData(
        data={
            FV.TXYH: np.zeros((2, 2, 3), dtype=float),
            FV.X: np.zeros((2, 2), dtype=float),
            "out_v": np.zeros((2, 2), dtype=float),
        },
        dims={
            FV.TXYH: (FC.STATE, FC.TURBINE, FC.XYH),
            FV.X: (FC.STATE, FC.TURBINE),
            "out_v": (FC.STATE, FC.TURBINE),
        },
    )

    out = model.calculate(algo=algo, mdata={}, fdata=fdata, st_sel=st_sel)

    expected = np.zeros((2, 2), dtype=float)
    expected[0, 0] = 10.0
    expected[1, 1] = 40.0
    assert np.allclose(out["out_v"], expected)


def main():
    test_rotor_centre_calc_extracts_selected_values()


if __name__ == "__main__":
    main()
