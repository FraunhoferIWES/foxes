import pytest

from _model_smoke_helpers import run_model_smoke


MODEL_PATHS = [
    "foxes.models.wake_deflections.bastankhah2016:Bastankhah2016Deflection",
    "foxes.models.wake_deflections.jimenez:JimenezDeflection",
    "foxes.models.wake_deflections.no_deflection:NoDeflection",
]


@pytest.mark.parametrize(
    "model_path",
    MODEL_PATHS,
    ids=lambda model_path: model_path.split(":")[1],
)
def test_wake_deflections_smoke(model_path, tmp_path):
    run_model_smoke(model_path, tmp_path)
