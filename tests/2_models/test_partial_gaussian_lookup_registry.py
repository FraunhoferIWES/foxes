from __future__ import annotations

import foxes

from _model_smoke_helpers import _assert_farm_results
from _model_smoke_helpers import _engine
from _model_smoke_helpers import _farm
from _model_smoke_helpers import _mbook_with_ttype


def test_model_book_registers_gaussian_lookup_partial_wakes():
    mbook, type_alias = _mbook_with_ttype()
    assert "gaussian_lookup" in mbook.partial_wakes
    assert (
        mbook.default_partial_wakes(mbook.wake_models["Bastankhah2014"])
        == "gaussian_lookup"
    )
    assert (
        mbook.default_partial_wakes(mbook.wake_models["Jensen_linear_k0075"])
        == "top_hat"
    )

    farm = _farm([type_alias])
    states = foxes.input.states.SingleStateStates(ws=8.0, wd=270.0, ti=0.08, rho=1.225)
    wake_alias = "Bastankhah2014_linear_k004"

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model="centre",
        wake_models=[wake_alias],
        partial_wakes={wake_alias: "gaussian_lookup"},
        mbook=mbook,
        verbosity=0,
    )

    with _engine():
        farm_results = algo.calc_farm()

    _assert_farm_results(farm_results)
    lookup_model = mbook.partial_wakes["gaussian_lookup"]
    assert lookup_model.lookup_dataset is not None
