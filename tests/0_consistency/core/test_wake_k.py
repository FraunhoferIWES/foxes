"""Wake-growth coefficient evaluation and removed-input diagnostics."""

import numpy as np
import pytest

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import FData, WakeK, WakeModel
from foxes.models.turbine_models import kTI


@pytest.mark.parametrize("ti_var", [FV.TI, FV.AMB_TI])
@pytest.mark.parametrize("selection", [None, np.array([[True, False], [False, True]])])
def test_linear_growth_uses_requested_turbulence_and_preserves_model(ti_var, selection):
    model = WakeK(k_0=0.01, k_1=0.3, ti_var=ti_var)
    fdata = FData.from_sizes(2, 2)
    turbulence = np.array([[0.1, 0.2], [0.3, 0.4]])
    fdata.add(ti_var, turbulence, (FC.STATE, FC.TURBINE))
    before = vars(model).copy()

    actual = model(FC.STATE_TURBINE, fdata=fdata, lookup_ti="f", selection=selection)

    expected = 0.01 + 0.3 * turbulence
    np.testing.assert_allclose(
        actual, expected if selection is None else expected[selection]
    )
    assert vars(model) == before


@pytest.mark.parametrize("coefficients", [{"k_0": 0.04}, {"k_0": 0.04, "k_1": 0.0}])
def test_intercept_only_growth_does_not_require_turbulence(coefficients):
    actual = WakeK(**coefficients)(FC.STATE_TURBINE, fdata=FData.from_sizes(2, 3))
    assert actual.shape == (2, 3)
    np.testing.assert_allclose(actual, 0.04)


def test_data_driven_growth_and_failed_ti_lookup_preserve_model():
    fdata = FData.from_sizes(2, 3)
    fdata.add(FV.K, np.full((2, 3), 0.07), (FC.STATE, FC.TURBINE))
    np.testing.assert_allclose(
        WakeK()(FC.STATE_TURBINE, fdata=fdata, lookup_k="sf"), 0.07
    )
    model = WakeK(k_0=0.01, k_1=0.3)
    before = vars(model).copy()
    with pytest.raises(ValueError, match="TI.*requested but not found"):
        model(FC.STATE_TURBINE, fdata=fdata, lookup_ti="f")
    assert vars(model) == before


@pytest.mark.parametrize("lookup", ["f", "fs"])
def test_growth_lookup_can_prefer_farm_data_over_linear_configuration(lookup):
    fdata = FData.from_sizes(2, 1)
    fdata.add(FV.K, np.full((2, 1), 0.07), (FC.STATE, FC.TURBINE))
    result = WakeK(k_0=0.01, k_1=0.3)(FC.STATE_TURBINE, fdata=fdata, lookup_k=lookup)
    np.testing.assert_allclose(result, 0.07)


@pytest.mark.parametrize("coefficient", ["k_0", "k_1"])
def test_uniform_growth_and_linear_coefficients_are_exclusive(coefficient):
    with pytest.raises(ValueError, match="Got 'k'"):
        WakeK(k=0.04, **{coefficient: 0.2})


@pytest.mark.parametrize("keyword", ["ka", "kb", "kti", "kTI"])
@pytest.mark.parametrize(
    "factory",
    [
        WakeK,
        kTI,
        lambda **kw: WakeModel.new("JensenWake", superposition="ws_linear", **kw),
        lambda **kw: WakeModel.new("IECTIWake", superposition="ti_linear", **kw),
    ],
)
def test_removed_coefficients_raise_migration_error(factory, keyword):
    with pytest.raises(DeprecationWarning, match=r"k = k_0 \+ k_1 \* TI"):
        factory(**{keyword: 0.2})


def test_iec_opening_angle_rejects_explicit_coefficients():
    with pytest.raises(KeyError, match="k_0.*k_1"):
        WakeModel.new("IECTIWake", superposition="ti_linear", k_0=0.04)
