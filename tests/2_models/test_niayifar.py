from __future__ import annotations

import foxes
import foxes.variables as FV
from foxes.input.yaml.windio.read_attributes import _read_wind_deficit
from foxes.models.wake_models.wind import Bastankhah2014
from foxes.utils import Dict


def test_niayifar_model_book_defaults():
    wake_model = foxes.models.ModelBook().wake_models["Niayifar"]

    assert isinstance(wake_model, Bastankhah2014)
    assert wake_model.wind_superposition == "ws_linear"
    assert wake_model.wake_k._ka == 0.3837
    assert wake_model.wake_k._kb == 0.003678
    assert wake_model.wake_k.ti_var == FV.TI


def test_niayifar_windio_defaults():
    mbook = foxes.models.ModelBook()
    algo_dict: dict[str, list[str]] = {"wake_models": []}

    _read_wind_deficit(
        "wind_deficit_model",
        Dict({"name": "Niayifar"}),
        Dict({"ws_superposition": "Linear"}),
        "Madsen",
        algo_dict,
        mbook,
        verbosity=0,
    )

    wake_model = mbook.wake_models["Niayifar"]
    assert isinstance(wake_model, Bastankhah2014)
    assert wake_model.wake_k._ka == 0.3837
    assert wake_model.wake_k._kb == 0.003678
    assert wake_model.wake_k.ti_var == FV.TI
    assert algo_dict["wake_models"] == ["Niayifar"]


def test_niayifar_windio_coefficient_convention():
    mbook = foxes.models.ModelBook()
    algo_dict: dict[str, list[str]] = {"wake_models": []}

    _read_wind_deficit(
        "wind_deficit_model",
        Dict(
            {
                "name": "Niayifar",
                "wake_expansion_coefficient": {"k_a": 0.01, "k_b": 0.2},
            }
        ),
        Dict({"ws_superposition": "Linear"}),
        "Madsen",
        algo_dict,
        mbook,
        verbosity=0,
    )

    wake_model = mbook.wake_models["Niayifar"]
    assert wake_model.wake_k._ka == 0.2
    assert wake_model.wake_k._kb == 0.01
