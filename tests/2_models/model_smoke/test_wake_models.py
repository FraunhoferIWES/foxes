import pytest

from _model_smoke_helpers import run_model_smoke


MODEL_PATHS = [
    "foxes.models.wake_models.induction.rankine_half_body:RankineHalfBody",
    "foxes.models.wake_models.induction.rathmann:Rathmann",
    "foxes.models.wake_models.induction.self_similar:SelfSimilar",
    "foxes.models.wake_models.induction.self_similar2020:SelfSimilar2020",
    "foxes.models.wake_models.induction.vortex_sheet:VortexSheet",
    "foxes.models.wake_models.ti.crespo_hernandez:CrespoHernandezTIWake",
    "foxes.models.wake_models.ti.iec_ti:IECTIWake",
    "foxes.models.wake_models.wind.bastankhah14:Bastankhah2014",
    "foxes.models.wake_models.wind.bastankhah16:Bastankhah2016",
    "foxes.models.wake_models.wind.bastankhah16:Bastankhah2016Model",
    "foxes.models.wake_models.wind.jensen:JensenWake",
    "foxes.models.wake_models.wind.jensen:JensenTurbOParkWake",
    "foxes.models.wake_models.wind.turbopark:TurbOParkWake",
    "foxes.models.wake_models.wind.turbopark:TurbOParkWakeIX",
]


@pytest.mark.parametrize(
    "model_path",
    MODEL_PATHS,
    ids=lambda model_path: model_path.split(":")[1],
)
def test_wake_models_smoke(model_path, tmp_path):
    run_model_smoke(model_path, tmp_path)
