import numpy as np
from types import SimpleNamespace

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import FData
from foxes.models.turbine_models.yaw2yawm import YAW2YAWM
from foxes.models.turbine_models.yawm2yaw import YAWM2YAW


def test_yaw2yawm_wraparound():
    model = YAW2YAWM()
    algo = SimpleNamespace(store_model_data=lambda *args, **kwargs: None)
    model.initialize(algo=algo, verbosity=0)

    st_sel = np.array([[True, True]])
    fdata = FData(
        data={
            FV.WD: np.array([[350.0, 10.0]]),
            FV.YAW: np.array([[10.0, 350.0]]),
            FV.YAWM: np.zeros((1, 2), dtype=float),
        },
        dims={
            FV.WD: (FC.STATE, FC.TURBINE),
            FV.YAW: (FC.STATE, FC.TURBINE),
            FV.YAWM: (FC.STATE, FC.TURBINE),
        },
    )

    out = model.calculate(algo=algo, mdata={}, fdata=fdata, st_sel=st_sel)
    assert np.allclose(out[FV.YAWM], np.array([[20.0, -20.0]]))


def test_yawm2yaw_wraparound():
    model = YAWM2YAW()
    algo = SimpleNamespace(store_model_data=lambda *args, **kwargs: None)
    model.initialize(algo=algo, verbosity=0)

    st_sel = np.array([[True, True]])
    fdata = FData(
        data={
            FV.WD: np.array([[350.0, 10.0]]),
            FV.YAWM: np.array([[20.0, -20.0]]),
            FV.YAW: np.zeros((1, 2), dtype=float),
        },
        dims={
            FV.WD: (FC.STATE, FC.TURBINE),
            FV.YAWM: (FC.STATE, FC.TURBINE),
            FV.YAW: (FC.STATE, FC.TURBINE),
        },
    )

    out = model.calculate(algo=algo, mdata={}, fdata=fdata, st_sel=st_sel)
    assert np.allclose(out[FV.YAW], np.array([[10.0, 350.0]]))


def main():
    test_yaw2yawm_wraparound()
    test_yawm2yaw_wraparound()


if __name__ == "__main__":
    main()
