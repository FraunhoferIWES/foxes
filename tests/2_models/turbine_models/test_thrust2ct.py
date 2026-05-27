import numpy as np
from types import SimpleNamespace

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import FData
from foxes.models.turbine_models.thrust2ct import Thrust2Ct


def test_thrust2ct_formula():
    model = Thrust2Ct()
    algo = SimpleNamespace(store_model_data=lambda *args, **kwargs: None)
    model.initialize(algo=algo, verbosity=0)

    st_sel = np.array([[True]])
    d = 100.0
    rho = 1.2
    ws = 8.0
    area = np.pi * (d / 2) ** 2
    target_ct = 0.7
    thrust = 0.5 * rho * area * ws**2 * target_ct

    fdata = FData(
        data={
            FV.CT: np.zeros((1, 1), dtype=float),
            FV.T: np.array([[thrust]], dtype=float),
            FV.RHO: np.array([[rho]], dtype=float),
            FV.D: np.array([[d]], dtype=float),
            FV.REWS2: np.array([[ws]], dtype=float),
        },
        dims={
            FV.CT: (FC.STATE, FC.TURBINE),
            FV.T: (FC.STATE, FC.TURBINE),
            FV.RHO: (FC.STATE, FC.TURBINE),
            FV.D: (FC.STATE, FC.TURBINE),
            FV.REWS2: (FC.STATE, FC.TURBINE),
        },
    )

    out = model.calculate(algo=algo, mdata={}, fdata=fdata, st_sel=st_sel)
    assert np.allclose(out[FV.CT], np.array([[target_ct]]))


def main():
    test_thrust2ct_formula()


if __name__ == "__main__":
    main()
