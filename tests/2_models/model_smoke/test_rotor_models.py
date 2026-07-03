import pytest

from _model_smoke_helpers import run_model_smoke


MODEL_PATHS = [
    "foxes.models.rotor_models.centre:CentreRotor",
    "foxes.models.rotor_models.direct_infusion:DirectMDataInfusion",
    "foxes.models.rotor_models.grid:GridRotor",
    "foxes.models.rotor_models.levels:LevelRotor",
]


@pytest.mark.parametrize(
    "model_path",
    MODEL_PATHS,
    ids=lambda model_path: model_path.split(":")[1],
)
def test_rotor_models_smoke(model_path, tmp_path):
    run_model_smoke(model_path, tmp_path)
