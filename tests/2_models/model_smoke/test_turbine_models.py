import pytest

from _model_smoke_helpers import run_model_smoke


MODEL_PATHS = [
    "foxes.models.turbine_models.calculator:Calculator",
    "foxes.models.turbine_models.kTI_model:kTI",
    "foxes.models.turbine_models.lookup_table:LookupTable",
    "foxes.models.turbine_models.power_mask:PowerMask",
    "foxes.models.turbine_models.rotor_centre_calc:RotorCentreCalc",
    "foxes.models.turbine_models.sector_management:SectorManagement",
    "foxes.models.turbine_models.set_farm_vars:SetFarmVars",
    "foxes.models.turbine_models.table_factors:TableFactors",
    "foxes.models.turbine_models.thrust2ct:Thrust2Ct",
    "foxes.models.turbine_models.yaw2yawm:YAW2YAWM",
    "foxes.models.turbine_models.yawcontroller:YawController",
    "foxes.models.turbine_models.yawm2yaw:YAWM2YAW",
]


@pytest.mark.parametrize(
    "model_path",
    MODEL_PATHS,
    ids=lambda model_path: model_path.split(":")[1],
)
def test_turbine_models_smoke(model_path, tmp_path):
    run_model_smoke(model_path, tmp_path)
