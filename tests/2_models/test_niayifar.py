from __future__ import annotations

import numpy as np
import pytest

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.core import FData
from foxes.input.yaml.windio.read_attributes import _read_wind_deficit, _read_turbulence
from foxes.models.wake_models.wind import Bastankhah2014
from foxes.utils import Dict


def test_niayifar_model_book_defaults():
    wake_model = foxes.models.ModelBook().wake_models["Niayifar"]

    assert isinstance(wake_model, Bastankhah2014)
    assert wake_model.wind_superposition == "ws_linear"
    assert wake_model.wake_k._k_1 == 0.3837
    assert wake_model.wake_k._k_0 == 0.003678
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
    assert wake_model.wake_k._k_1 == 0.3837
    assert wake_model.wake_k._k_0 == 0.003678
    assert wake_model.wake_k.ti_var == FV.TI
    assert algo_dict["wake_models"] == ["Niayifar"]


@pytest.mark.parametrize("ambient", [False, True])
@pytest.mark.parametrize("slope", [0.0, 0.2])
def test_niayifar_windio_coefficient_convention(ambient, slope):
    mbook = foxes.models.ModelBook()
    algo_dict: dict[str, list[str]] = {"wake_models": []}

    _read_wind_deficit(
        "wind_deficit_model",
        Dict(
            {
                "name": "Niayifar",
                "wake_expansion_coefficient": {
                    "k_a": slope,
                    "k_b": 0.01,
                    "free_stream_ti": ambient,
                },
            }
        ),
        Dict({"ws_superposition": "Linear"}),
        "Madsen",
        algo_dict,
        mbook,
        verbosity=0,
    )

    growth = mbook.wake_models["Niayifar"].wake_k
    fdata = FData.from_sizes(2, 1)
    if slope != 0:
        ti_var = FV.AMB_TI if ambient else FV.TI
        fdata.add(ti_var, np.array([[0.1], [0.2]]), (FC.STATE, FC.TURBINE))
    actual = growth(FC.STATE_TURBINE, fdata=fdata, lookup_ti="f")
    np.testing.assert_allclose(
        actual, np.array([[0.01 + slope * 0.1], [0.01 + slope * 0.2]])
    )


@pytest.mark.parametrize("ambient", [False, True])
@pytest.mark.parametrize("override", [None, {"k_a": 0.4, "k_b": 0.02}])
def test_windio_turbulence_translates_inherited_and_explicit_coefficients(
    ambient, override
):
    mbook = foxes.models.ModelBook()
    configuration = {"name": "IEC-TI-2019"}
    if override is not None:
        configuration["wake_expansion_coefficient"] = {
            **override,
            "free_stream_ti": ambient,
        }
    _read_turbulence(
        Dict(configuration),
        Dict({"ti_superposition": "Squared"}),
        "Madsen",
        {"wake_models": []},
        mbook,
        0.01,
        0.2,
        ambient,
        verbosity=0,
    )
    growth = mbook.wake_models["IEC-TI-2019"].wake_k
    ti_var = FV.AMB_TI if ambient else FV.TI
    fdata = FData.from_sizes(2, 1)
    fdata.add(ti_var, np.array([[0.1], [0.2]]), (FC.STATE, FC.TURBINE))

    actual = growth(FC.STATE_TURBINE, fdata=fdata, lookup_ti="f")

    intercept, multiplier = (0.01, 0.2) if override is None else (0.02, 0.4)
    np.testing.assert_allclose(actual, intercept + multiplier * fdata[ti_var])
    assert growth.ti_var == ti_var
