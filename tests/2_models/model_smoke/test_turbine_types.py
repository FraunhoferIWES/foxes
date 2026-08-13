import pytest

from _model_smoke_helpers import run_model_smoke


MODEL_PATHS = [
    "foxes.models.turbine_types.CpCt_file:CpCtFile",
    "foxes.models.turbine_types.CpCt_from_two:CpCtFromTwo",
    "foxes.models.turbine_types.PCt_file:PCtFile",
    "foxes.models.turbine_types.PCt_from_two:PCtFromTwo",
    "foxes.models.turbine_types.TBL_file:TBLFile",
    "foxes.models.turbine_types.calculator_type:CalculatorType",
    "foxes.models.turbine_types.lookup:FromLookupTable",
    "foxes.models.turbine_types.null_type:NullType",
    "foxes.models.turbine_types.wsrho2PCt_from_two:WsRho2PCtFromTwo",
    "foxes.models.turbine_types.wsti2PCt_from_two:WsTI2PCtFromTwo",
]


@pytest.mark.parametrize(
    "model_path",
    MODEL_PATHS,
    ids=lambda model_path: model_path.split(":")[1],
)
def test_turbine_types_smoke(model_path, tmp_path):
    run_model_smoke(model_path, tmp_path)
