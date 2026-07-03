import pytest

from _model_smoke_helpers import run_model_smoke


MODEL_PATHS = [
    "foxes.models.wake_frames.dynamic_wakes:DynamicWakes",
    "foxes.models.wake_frames.farm_order:FarmOrder",
    "foxes.models.wake_frames.rotor_wd:RotorWD",
    "foxes.models.wake_frames.seq_dynamic_wakes:SeqDynamicWakes",
    "foxes.models.wake_frames.streamlines:Streamlines2D",
    "foxes.models.wake_frames.timelines:Timelines",
]


@pytest.mark.parametrize(
    "model_path",
    MODEL_PATHS,
    ids=lambda model_path: model_path.split(":")[1],
)
def test_wake_frames_smoke(model_path, tmp_path):
    run_model_smoke(model_path, tmp_path)
