import pytest

from _model_smoke_helpers import run_model_smoke


MODEL_PATHS = [
    "foxes.models.wake_superpositions.ti_linear:TILinear",
    "foxes.models.wake_superpositions.ti_max:TIMax",
    "foxes.models.wake_superpositions.ti_pow:TIPow",
    "foxes.models.wake_superpositions.ti_quadratic:TIQuadratic",
    "foxes.models.wake_superpositions.wind_vector:WindVectorLinear",
    "foxes.models.wake_superpositions.ws_linear:WSLinear",
    "foxes.models.wake_superpositions.ws_linear:WSLinearLocal",
    "foxes.models.wake_superpositions.ws_max:WSMax",
    "foxes.models.wake_superpositions.ws_max:WSMaxLocal",
    "foxes.models.wake_superpositions.ws_pow:WSPow",
    "foxes.models.wake_superpositions.ws_pow:WSPowLocal",
    "foxes.models.wake_superpositions.ws_product:WSProduct",
    "foxes.models.wake_superpositions.ws_quadratic:WSQuadratic",
    "foxes.models.wake_superpositions.ws_quadratic:WSQuadraticLocal",
]


@pytest.mark.parametrize(
    "model_path",
    MODEL_PATHS,
    ids=lambda model_path: model_path.split(":")[1],
)
def test_wake_superpositions_smoke(model_path, tmp_path):
    run_model_smoke(model_path, tmp_path)
