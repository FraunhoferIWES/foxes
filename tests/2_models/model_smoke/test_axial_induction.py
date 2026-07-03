import pytest

from _model_smoke_helpers import run_model_smoke


MODEL_PATHS = [
    "foxes.models.axial_induction.betz:BetzAxialInduction",
    "foxes.models.axial_induction.madsen:MadsenAxialInduction",
]


@pytest.mark.parametrize(
    "model_path",
    MODEL_PATHS,
    ids=lambda model_path: model_path.split(":")[1],
)
def test_axial_induction_smoke(model_path, tmp_path):
    run_model_smoke(model_path, tmp_path)
