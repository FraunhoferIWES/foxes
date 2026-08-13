import pytest

from _model_smoke_helpers import run_model_smoke


MODEL_PATHS = [
    "foxes.models.point_models.set_uniform_data:SetUniformData",
    "foxes.models.point_models.tke2ti:TKE2TI",
    "foxes.models.point_models.ustar2ti:Ustar2TI",
    "foxes.models.point_models.wake_deltas:WakeDeltas",
]


@pytest.mark.parametrize(
    "model_path",
    MODEL_PATHS,
    ids=lambda model_path: model_path.split(":")[1],
)
def test_point_models_smoke(model_path, tmp_path):
    run_model_smoke(model_path, tmp_path)
