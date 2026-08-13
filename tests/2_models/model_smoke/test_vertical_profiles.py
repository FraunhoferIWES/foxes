import pytest

from _model_smoke_helpers import run_model_smoke


MODEL_PATHS = [
    "foxes.models.vertical_profiles.abl_log_neutral_ws:ABLLogNeutralWsProfile",
    "foxes.models.vertical_profiles.abl_log_stable_ws:ABLLogStableWsProfile",
    "foxes.models.vertical_profiles.abl_log_unstable_ws:ABLLogUnstableWsProfile",
    "foxes.models.vertical_profiles.abl_log_ws:ABLLogWsProfile",
    "foxes.models.vertical_profiles.data_profile:DataProfile",
    "foxes.models.vertical_profiles.sheared_ws:ShearedProfile",
    "foxes.models.vertical_profiles.uniform:UniformProfile",
]


@pytest.mark.parametrize(
    "model_path",
    MODEL_PATHS,
    ids=lambda model_path: model_path.split(":")[1],
)
def test_vertical_profiles_smoke(model_path, tmp_path):
    run_model_smoke(model_path, tmp_path)
