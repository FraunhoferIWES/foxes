import numpy as np
import pytest
from types import SimpleNamespace

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import FData
from foxes.models.axial_induction import BetzAxialInduction
from foxes.models.turbine_models.power_mask import PowerMask
from foxes.utils import cubic_roots


class DummyInduction:
    initialized = False
    running = False
    name = "DummyInduction"

    def initialize(self, algo, loaded_data=None, force=False, verbosity=0):
        self.initialized = True
        return loaded_data

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
            FV.REWS: np.array([[9.0]], dtype=float),
            FV.RHO: np.array([[1.225]], dtype=float),
            FV.D: np.array([[100.0]], dtype=float),
            FV.CT: np.array([[ct0]], dtype=float),
        },
        dims={
            FV.P: (FC.STATE, FC.TURBINE),
            FV.MAX_P: (FC.STATE, FC.TURBINE),
            FV.REWS: (FC.STATE, FC.TURBINE),
            FV.RHO: (FC.STATE, FC.TURBINE),
            FV.D: (FC.STATE, FC.TURBINE),
            FV.CT: (FC.STATE, FC.TURBINE),
        },
    )

    out = model.calculate(algo=algo, mdata={}, fdata=fdata, st_sel=st_sel)

    ws = fdata[FV.REWS][st_sel]
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


def _calculate_boost(power, max_power, st_sel=slice(None), **kwargs):
    power, max_power = np.broadcast_arrays(
        np.asarray(power, dtype=float), np.asarray(max_power, dtype=float)
    )
    model = PowerMask(induction=BetzAxialInduction(), **kwargs)
    algo = SimpleNamespace(
        store_model_data=lambda *args, **kwargs: None,
        farm_controller=SimpleNamespace(
            turbine_types=[SimpleNamespace(P_nominal=2000.0, name="T0")]
        ),
    )
    model.initialize(algo=algo, verbosity=0)
    data = {
        FV.P: power[:, None].copy(),
        FV.MAX_P: max_power[:, None].copy(),
        FV.REWS: np.full((power.size, 1), 9.0),
        FV.RHO: np.full((power.size, 1), 1.225),
        FV.D: np.full((power.size, 1), 100.0),
        FV.CT: np.full((power.size, 1), 0.64),
    }
    fdata = FData(data=data, dims={name: (FC.STATE, FC.TURBINE) for name in data})
    return model.calculate(algo=algo, mdata={}, fdata=fdata, st_sel=st_sel)


def test_power_mask_boost_is_continuous_at_onset_and_rated_power():
    power = np.array([1899.9999, 1900.0001, 1999.9999, 2000.0001])
    out = _calculate_boost(power, 2100.0)

    assert abs(out[FV.P][1, 0] - out[FV.P][0, 0]) < 0.001
    assert abs(out[FV.CT][1, 0] - out[FV.CT][0, 0]) < 0.001
    assert abs(out[FV.P][3, 0] - out[FV.P][2, 0]) < 0.001
    assert abs(out[FV.CT][3, 0] - out[FV.CT][2, 0]) < 0.001


def test_power_mask_boost_thrust_saturates_at_physical_induction_limit():
    max_power = np.array([2100.0, 2300.0, 2314.8148, 2314.8149, 2600.0])
    out = _calculate_boost(np.full(5, 2000.0), max_power)

    np.testing.assert_allclose(out[FV.P][:, 0], max_power)
    assert np.all(np.isfinite(out[FV.CT]))
    assert np.all((out[FV.CT] >= 0.0) & (out[FV.CT] <= 8.0 / 9.0 + 1e-8))
    assert np.all(np.diff(out[FV.CT][:, 0]) >= -1e-8)
    np.testing.assert_allclose(out[FV.CT][-2:, 0], 8.0 / 9.0, atol=1e-7)


@pytest.mark.parametrize("width", [-1.0, np.nan, np.inf])
def test_power_mask_rejects_invalid_transition_width(width):
    with pytest.raises(ValueError, match="P_lim must be finite and nonnegative"):
        PowerMask(P_lim=width)


def test_power_mask_preserves_nan_masks_and_unselected_entries():
    selection = np.array([[True], [False], [True]])
    out = _calculate_boost(
        [2000.0, 2000.0, 1950.0], [np.nan, 2100.0, 2100.0], st_sel=selection
    )

    np.testing.assert_allclose(out[FV.P][:, 0], [2000.0, 2000.0, 2025.0])
    np.testing.assert_allclose(out[FV.CT][:2, 0], 0.64)
    assert 0.64 < out[FV.CT][2, 0] < 8.0 / 9.0


def test_power_mask_zero_width_and_shutdown():
    out = _calculate_boost([1999.0, 2000.0, 2001.0], 2100.0, P_lim=0.0)
    np.testing.assert_allclose(out[FV.P][:, 0], [1999.0, 2000.0, 2100.0])

    shutdown = _calculate_boost([2000.0], 0.0)
    np.testing.assert_allclose(shutdown[FV.P], 0.0)
    np.testing.assert_allclose(shutdown[FV.CT], 0.0)


def main():
    test_power_mask_applies_limit()


if __name__ == "__main__":
    main()
