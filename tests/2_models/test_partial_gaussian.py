from __future__ import annotations

import foxes
import foxes.variables as FV
import numpy as np
import pytest

from _model_smoke_helpers import _engine
from _model_smoke_helpers import _farm
from _model_smoke_helpers import _mbook_with_ttype
from foxes.models.partial_wakes.gaussian import PartialGaussian


def _calc_rews(partial_wakes: str) -> np.ndarray:
    mbook, type_alias = _mbook_with_ttype()
    farm = _farm([type_alias])
    states = foxes.input.states.SingleStateStates(
        ws=8.0,
        wd=270.0,
        ti=0.08,
        rho=1.225,
    )
    wake_model = "Bastankhah2014_linear_k004"
    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model="centre",
        wake_models=[wake_model],
        partial_wakes={wake_model: partial_wakes},
        mbook=mbook,
        verbosity=0,
    )
    with _engine():
        return algo.calc_farm()[FV.REWS].to_numpy()


def test_partial_gaussian_is_registered_and_matches_lookup():
    analytical = _calc_rews("gaussian")
    lookup = _calc_rews("gaussian_lookup")

    np.testing.assert_allclose(analytical, lookup, rtol=2.0e-3, atol=1.0e-5)


def test_partial_gaussian_rejects_non_gaussian_wake_model():
    model = PartialGaussian()
    model.name = "gaussian"
    wmodel = foxes.models.ModelBook().wake_models["Jensen_linear_k007"]

    with pytest.raises(TypeError, match="GaussianWakeModel"):
        model.check_wmodel(wmodel, error=True)


def test_partial_gaussian_rejects_negative_min_weight():
    with pytest.raises(ValueError, match="min_weight"):
        PartialGaussian(min_weight=-1.0)
