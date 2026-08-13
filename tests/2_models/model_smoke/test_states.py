import pytest

from _model_smoke_helpers import run_model_smoke


MODEL_PATHS = [
    "foxes.input.states.dataset_states:DatasetStates",
    "foxes.input.states.field_data:FieldData",
    "foxes.input.states.field_data:LatLonFieldData",
    "foxes.input.states.field_data:WeibullField",
    "foxes.input.states.icon_states:ICONStates",
    "foxes.input.states.multi_height:MultiHeightNCStates",
    "foxes.input.states.multi_height:MultiHeightNCTimeseries",
    "foxes.input.states.multi_height:MultiHeightStates",
    "foxes.input.states.multi_height:MultiHeightTimeseries",
    "foxes.input.states.newa_states:NEWAStates",
    "foxes.input.states.one_point_flow:OnePointFlowMultiHeightNCTimeseries",
    "foxes.input.states.one_point_flow:OnePointFlowMultiHeightTimeseries",
    "foxes.input.states.one_point_flow:OnePointFlowStates",
    "foxes.input.states.one_point_flow:OnePointFlowTimeseries",
    "foxes.input.states.point_cloud_data:PointCloudData",
    "foxes.input.states.point_cloud_data:TurbinePointCloud",
    "foxes.input.states.point_cloud_data:WeibullPointCloud",
    "foxes.input.states.scan:ScanStates",
    "foxes.input.states.single:SingleStateStates",
    "foxes.input.states.single_state_field:SingleStateField",
    "foxes.input.states.states_table:StatesTable",
    "foxes.input.states.states_table:TabStates",
    "foxes.input.states.states_table:Timeseries",
    "foxes.input.states.weibull_sectors:WeibullSectors",
    "foxes.input.states.wrg_states:WRGStates",
]


@pytest.mark.parametrize(
    "model_path",
    MODEL_PATHS,
    ids=lambda model_path: model_path.split(":")[1],
)
def test_states_smoke(model_path, tmp_path):
    run_model_smoke(model_path, tmp_path)
