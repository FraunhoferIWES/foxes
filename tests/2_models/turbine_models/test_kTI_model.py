import numpy as np
import pytest
from types import SimpleNamespace

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import FData
from foxes.models.turbine_models.kTI_model import kTI


def test_kti_calculate_selected_entries():
    model = kTI(k_1=0.3, k_0=0.01, ti_var=FV.TI, k_var=FV.K)
    algo = SimpleNamespace(store_model_data=lambda *args, **kwargs: None)
    model.initialize(algo=algo, verbosity=0)

    st_sel = np.array([[True, False], [True, True]])
    fdata = FData(
        data={FV.K: np.zeros((2, 2), dtype=float)},
        dims={FV.K: (FC.STATE, FC.TURBINE)},
    )

    ti_sel = np.array([0.1, 0.2, 0.3], dtype=float)

    def fake_get_data(v, *_args, **_kwargs):
        if v == FV.K_1:
            return np.full(ti_sel.shape, 0.3)
        if v == FV.K_0:
            return np.full(ti_sel.shape, 0.01)
        if v == FV.TI:
            return ti_sel
        raise KeyError(v)

    model.get_data = fake_get_data

    out = model.calculate(algo=algo, mdata={}, fdata=fdata, st_sel=st_sel)

    expected = np.zeros((2, 2), dtype=float)
    expected[st_sel] = 0.3 * ti_sel + 0.01
    assert np.allclose(out[FV.K], expected)


@pytest.mark.parametrize("selection", [slice(None), slice(1, None), slice(0, 0)])
def test_kti_uniform_inputs_support_state_slices(selection):
    model = kTI(k_1=0.3, k_0=0.01, ti_val=0.2)
    fdata = FData.from_sizes(3, 2)
    fdata.add(FV.K, np.zeros((3, 2)), (FC.STATE, FC.TURBINE))

    out = model.calculate(algo=None, mdata={}, fdata=fdata, st_sel=selection)

    expected = np.zeros((3, 2))
    expected[selection] = 0.07
    np.testing.assert_allclose(out[FV.K], expected)


@pytest.mark.parametrize("ti_var", [FV.TI, FV.AMB_TI])
@pytest.mark.parametrize("uniform_intercept", [None, 0.02])
def test_kti_reads_state_turbine_coefficients(ti_var, uniform_intercept):
    model = kTI(k_0=uniform_intercept, ti_var=ti_var)
    fdata = FData.from_sizes(2, 2)
    fdata.add(FV.K, np.full((2, 2), 0.8), (FC.STATE, FC.TURBINE))
    intercept = np.array([[0.01, 0.02], [0.03, 0.04]])
    multiplier = np.array([[0.2, 0.3], [0.4, 0.5]])
    turbulence = np.array([[0.1, 0.15], [0.2, 0.25]])
    for variable, values in [
        (FV.K_0, intercept),
        (FV.K_1, multiplier),
        (ti_var, turbulence),
    ]:
        fdata.add(variable, values, (FC.STATE, FC.TURBINE))
    selection = np.array([[True, False], [False, True]])

    result = model.calculate(None, {}, fdata, st_sel=selection)[FV.K]

    expected = np.full((2, 2), 0.8)
    growth = (
        intercept if uniform_intercept is None else uniform_intercept
    ) + multiplier * turbulence
    expected[selection] = growth[selection]
    np.testing.assert_allclose(result, expected)


def test_kti_missing_intercept_defaults_to_zero_but_multiplier_is_required():
    fdata = FData.from_sizes(2, 2)
    result = kTI(k_1=0.3, ti_val=0.2).calculate(None, {}, fdata)
    np.testing.assert_allclose(result[FV.K], 0.06)
    with pytest.raises(ValueError, match="k1.*requested but not found"):
        kTI(ti_val=0.2).calculate(None, {}, fdata)


def main():
    test_kti_calculate_selected_entries()


if __name__ == "__main__":
    main()
