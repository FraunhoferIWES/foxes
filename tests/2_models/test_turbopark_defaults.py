from __future__ import annotations

import foxes
import foxes.variables as FV
from foxes.input.yaml.windio.read_attributes import _read_wind_deficit
from foxes.models.wake_models.wind import TurbOParkWake
from foxes.utils import Dict

from _model_smoke_helpers import _assert_farm_results
from _model_smoke_helpers import _engine
from _model_smoke_helpers import _farm
from _model_smoke_helpers import _mbook_with_ttype


def test_turbopark_defaults_match_original_model():
    mbook, turbine_type = _mbook_with_ttype()
    wake_model = mbook.wake_models["TurbOPark"]

    direct_wake_model = TurbOParkWake()
    assert direct_wake_model.induction == "Betz"
    assert direct_wake_model.wind_superposition == "ws_quadratic_amb_target"
    assert wake_model.induction == "Betz"
    assert wake_model.wind_superposition == "ws_quadratic_amb_target"
    assert wake_model.wake_k.repr() == f"k=0.04*{FV.AMB_TI}"
    assert mbook.default_partial_wakes(wake_model) == "gaussian"

    algo = foxes.algorithms.Downwind(
        _farm([turbine_type]),
        foxes.input.states.SingleStateStates(ws=8.0, wd=270.0, ti=0.08, rho=1.225),
        wake_models=["TurbOPark"],
        mbook=mbook,
        verbosity=0,
    )

    assert algo.ground_models["TurbOPark"].name == "no_ground"
    assert algo.partial_wakes["TurbOPark"].name == "gaussian"

    with _engine():
        farm_results = algo.calc_farm()
    _assert_farm_results(farm_results)


def test_windio_turbopark_uses_original_model_structure():
    mbook = foxes.models.ModelBook()
    algo_dict: dict[str, list[str]] = {"wake_models": []}
    wind_deficit = Dict(
        {
            "name": "TurbOPark",
            "wake_expansion_coefficient": {
                "k_a": 0.0,
                "k_b": 0.04,
                "free_stream_ti": True,
            },
        }
    )
    superposition = Dict({"ws_superposition": "Linear"})

    _read_wind_deficit(
        "wind_deficit_model",
        wind_deficit,
        superposition,
        "Madsen",
        algo_dict,
        mbook,
        verbosity=0,
    )

    wake_model = mbook.wake_models["TurbOPark"]
    assert wake_model.induction == "Betz"
    assert wake_model.wind_superposition == "ws_quadratic_amb_target"
    assert wake_model.wake_k.repr() == f"k=0.04*{FV.AMB_TI}"
    assert algo_dict["wake_models"] == ["TurbOPark"]
    assert algo_dict["ground_models"] == {"TurbOPark": "ground_mirror"}


def test_windio_keeps_non_turbopark_configuration():
    mbook = foxes.models.ModelBook()
    algo_dict: dict[str, list[str]] = {"wake_models": []}
    wind_deficit = Dict(
        {
            "name": "Jensen",
            "wake_expansion_coefficient": {"k_a": 0.04},
        }
    )
    superposition = Dict({"ws_superposition": "Linear"})

    _read_wind_deficit(
        "wind_deficit_model",
        wind_deficit,
        superposition,
        "Madsen",
        algo_dict,
        mbook,
        verbosity=0,
    )

    wake_model = mbook.wake_models["Jensen"]
    assert wake_model.induction == "Madsen"
    assert wake_model.wind_superposition == "ws_linear"


def test_wind_speed_superposition_scaling_aliases():
    mbook = foxes.models.ModelBook()
    expected = {
        "ws_linear": (False, False),
        "ws_linear_amb": (True, False),
        "ws_linear_target": (False, True),
        "ws_linear_amb_target": (True, True),
        "ws_quadratic": (False, False),
        "ws_quadratic_amb": (True, False),
        "ws_quadratic_target": (False, True),
        "ws_quadratic_amb_target": (True, True),
        "ws_cubic": (False, False),
        "ws_cubic_amb": (True, False),
        "ws_cubic_target": (False, True),
        "ws_cubic_amb_target": (True, True),
        "ws_quartic": (False, False),
        "ws_quartic_amb": (True, False),
        "ws_quartic_target": (False, True),
        "ws_quartic_amb_target": (True, True),
        "ws_max": (False, False),
        "ws_max_amb": (True, False),
        "ws_max_target": (False, True),
        "ws_max_amb_target": (True, True),
        "vector": (False, False),
        "vector_amb": (True, False),
        "vector_target": (False, True),
        "vector_amb_target": (True, True),
    }

    for name, (scale_amb, scale_target) in expected.items():
        superposition = mbook.wake_superpositions[name]
        assert superposition.scale_amb is scale_amb
        assert superposition.scale_target is scale_target
