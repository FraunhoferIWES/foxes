import pytest

from _model_smoke_helpers import run_model_smoke


MODEL_PATHS = [
    "foxes.models.partial_wakes.axiwake:PartialAxiwake",
    "foxes.models.partial_wakes.centre:PartialCentre",
    "foxes.models.partial_wakes.grid:PartialGrid",
    "foxes.models.partial_wakes.rotor_points:RotorPoints",
    "foxes.models.partial_wakes.segregated:PartialSegregated",
    "foxes.models.partial_wakes.top_hat:PartialTopHat",
]


@pytest.mark.parametrize(
    "model_path",
    MODEL_PATHS,
    ids=lambda model_path: model_path.split(":")[1],
)
def test_partial_wakes_smoke(model_path, tmp_path):
    run_model_smoke(model_path, tmp_path)
