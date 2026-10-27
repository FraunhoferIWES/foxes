from __future__ import annotations

import pytest

from foxes.input.yaml.windio.read_attributes import _read_blockage
from foxes.models import ModelBook
from foxes.utils import Dict


@pytest.mark.parametrize(
    "name",
    [
        "RankineHalfBody",
        "Rathmann",
        "SelfSimilarityDeficit",
        "SelfSimilarityDeficit2020",
    ],
)
@pytest.mark.parametrize("existing_ground_models", [False, True])
def test_blockage_enables_ground_mirror(name, existing_ground_models):
    algo_dict = {"algo_type": "Downwind", "wake_models": ["TurbOPark", "Jensen"]}
    expected_ground_models = {}
    if existing_ground_models:
        expected_ground_models = {"TurbOPark": "ground_mirror", "Jensen": "no_ground"}
        algo_dict["ground_models"] = expected_ground_models.copy()
    mbook = ModelBook()

    _read_blockage(Dict({"name": name}), "Betz", algo_dict, mbook, verbosity=0)

    assert algo_dict["ground_models"] == {
        **expected_ground_models,
        name: "ground_mirror",
    }
    assert algo_dict["wake_models"] == ["TurbOPark", "Jensen", name]
    assert algo_dict["algo_type"] == "Iterative"
    assert name in mbook.wake_models


@pytest.mark.parametrize("name", ["None", "none"])
def test_disabled_blockage_leaves_algorithm_unchanged(name):
    algo_dict = {"algo_type": "Downwind", "wake_models": []}

    _read_blockage(Dict({"name": name}), "Betz", algo_dict, ModelBook(), verbosity=0)

    assert algo_dict == {"algo_type": "Downwind", "wake_models": []}


def test_unknown_blockage_model_is_rejected():
    algo_dict = {"algo_type": "Downwind", "wake_models": []}

    with pytest.raises(KeyError, match="unsupported_blockage"):
        _read_blockage(
            Dict({"name": "unsupported_blockage"}),
            "Betz",
            algo_dict,
            ModelBook(),
            verbosity=0,
        )

    assert algo_dict == {"algo_type": "Downwind", "wake_models": []}
