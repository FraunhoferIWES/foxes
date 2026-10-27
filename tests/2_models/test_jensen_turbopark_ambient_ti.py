from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import foxes
import foxes.variables as FV
from foxes.models.wake_models.wind import JensenTurbOParkWake

from _model_smoke_helpers import _farm, _mbook_with_ttype


def _calculate(amb_ti=None, state_ti=None):
    mbook, turbine_type = _mbook_with_ttype()
    mbook.wake_models["test_wake"] = JensenTurbOParkWake(
        amb_ti=amb_ti, ka=0.6, ti_var=FV.AMB_TI
    )
    values = pd.DataFrame({FV.WS: [8.0] * 4, FV.WD: [270.0, 90.0, 0.0, 270.0]})
    output_vars = [FV.WS, FV.WD]
    if state_ti is not None:
        values[FV.TI] = state_ti
        output_vars.append(FV.TI)
    states = foxes.input.states.StatesTable(
        values, output_vars=output_vars, fixed_vars={FV.RHO: 1.225}
    )
    algo = foxes.algorithms.Downwind(
        _farm([turbine_type]),
        states,
        wake_models=["test_wake"],
        partial_wakes="top_hat",
        mbook=mbook,
        verbosity=0,
    )
    with foxes.Engine.new("numpy", chunk_size_states=3):
        return algo.calc_farm()


def test_constant_ambient_ti_matches_farm_data_without_adapters():
    constant = _calculate(amb_ti=0.06)
    from_states = _calculate(state_ti=0.06)

    for variable in (FV.P, FV.REWS):
        np.testing.assert_allclose(constant[variable], from_states[variable])
        assert np.isfinite(constant[variable]).all()
    assert FV.AMB_TI not in constant


def test_farm_ambient_ti_takes_precedence_over_constant():
    turbulence = [0.04, 0.08, 0.06, 0.1]
    reference = _calculate(state_ti=turbulence)
    with_constant = _calculate(amb_ti=0.2, state_ti=turbulence)

    np.testing.assert_allclose(with_constant[FV.P], reference[FV.P])
    np.testing.assert_allclose(with_constant[FV.AMB_TI], reference[FV.AMB_TI])


def test_missing_ambient_ti_raises_contextual_error():
    with pytest.raises(ValueError, match="AMB_TI.*requested but not found"):
        _calculate()


@pytest.mark.parametrize("amb_ti", [0.0, -0.01, np.nan, np.inf, -np.inf])
def test_invalid_constant_ambient_ti_is_rejected(amb_ti):
    with pytest.raises(ValueError, match="amb_ti must be finite and strictly positive"):
        JensenTurbOParkWake(amb_ti=amb_ti)
