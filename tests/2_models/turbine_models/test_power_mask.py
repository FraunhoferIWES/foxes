import numpy as np
from types import SimpleNamespace

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import FData
from foxes.models.turbine_models.power_mask import PowerMask
from foxes.utils import cubic_roots


class DummyInduction:
    initialized = False
    running = False
    name = "DummyInduction"

    def initialize(self, algo, verbosity=0, force=False):
        self.initialized = True

    def ct2a(self, ct):
        return np.full_like(ct, 0.2, dtype=float)


def test_power_mask_applies_limit():
    model = PowerMask(induction=DummyInduction(), factor_P=1.0)
    algo = SimpleNamespace(
        store_model_data=lambda *args, **kwargs: None,
        farm_controller=SimpleNamespace(
            turbine_types=[SimpleNamespace(P_nominal=2000.0, name="T0")]
        ),
    )
    model.initialize(algo=algo, verbosity=0)

    st_sel = np.array([[True]])
    p0 = 3000.0
    max_p0 = 1500.0
    ct0 = 0.8
    fdata = FData(
        data={
            FV.P: np.array([[p0]], dtype=float),
            FV.MAX_P: np.array([[max_p0]], dtype=float),
            FV.REWS3: np.array([[9.0]], dtype=float),
            FV.RHO: np.array([[1.225]], dtype=float),
            FV.D: np.array([[100.0]], dtype=float),
            FV.CT: np.array([[ct0]], dtype=float),
        },
        dims={
            FV.P: (FC.STATE, FC.TURBINE),
            FV.MAX_P: (FC.STATE, FC.TURBINE),
            FV.REWS3: (FC.STATE, FC.TURBINE),
            FV.RHO: (FC.STATE, FC.TURBINE),
            FV.D: (FC.STATE, FC.TURBINE),
            FV.CT: (FC.STATE, FC.TURBINE),
        },
    )

    out = model.calculate(algo=algo, mdata={}, fdata=fdata, st_sel=st_sel)

    ws = fdata[FV.REWS3][st_sel]
    rho = fdata[FV.RHO][st_sel]
    r = fdata[FV.D][st_sel] / 2
    denom = 0.5 * ws**3 * rho * np.pi * r**2
    cp = np.array([max_p0], dtype=float) / denom
    cp0 = np.array([p0], dtype=float) / denom
    a0 = DummyInduction().ct2a(np.array([ct0], dtype=float))
    cp_a0 = 4 * a0**3 - 8 * a0**2 + 4 * a0
    e = cp0 / cp_a0
    rts = cubic_roots(
        -cp / e,
        np.full(1, 4.0, dtype=float),
        np.full(1, -8.0, dtype=float),
        np.full(1, 4.0, dtype=float),
    )
    rts[np.isnan(rts)] = np.inf
    rts[rts <= 0.0] = np.inf
    a = np.min(rts, axis=1)
    expected_ct = 4 * a * (1 - a)

    assert np.allclose(out[FV.P], np.array([[1500.0]]))
    assert np.allclose(out[FV.CT][st_sel], expected_ct)


def main():
    test_power_mask_applies_limit()


if __name__ == "__main__":
    main()
