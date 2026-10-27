from __future__ import annotations

import pytest

from foxes.input.yaml.windio.read_attributes import _read_rotor_averaging
from foxes.utils import Dict


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Center", "centre"),
        ("None", "centre"),
        ("area_overlap", "top_hat"),
        ("gaussian_overlap", "gaussian_lookup"),
    ],
)
def test_rotor_averaging_name_sets_partial_wakes(name, expected):
    algo_dict = {}

    _read_rotor_averaging(Dict({"name": name}), algo_dict, verbosity=0)

    assert algo_dict["partial_wakes"] == expected


def test_wake_averaging_takes_precedence_over_name():
    algo_dict = {}
    rotor_averaging = Dict({"name": "gaussian_overlap", "wake_averaging": "center"})

    _read_rotor_averaging(rotor_averaging, algo_dict, verbosity=0)

    assert algo_dict["partial_wakes"] == "centre"


def test_unknown_rotor_averaging_name_uses_model_default():
    algo_dict = {}

    _read_rotor_averaging(
        Dict({"name": "unsupported_averaging"}), algo_dict, verbosity=0
    )

    assert algo_dict["partial_wakes"] is None


@pytest.mark.parametrize(
    ("background_averaging", "grid", "expected"),
    [
        ("grid", "grid", "rotor_points"),
        ("center", "grid", "grid4"),
        ("center", "grid9", "grid9"),
    ],
)
def test_grid_rotor_averaging_selects_partial_wakes(
    background_averaging, grid, expected
):
    algo_dict = {}
    rotor_averaging = Dict(
        {
            "name": "grid",
            "grid": grid,
            "n_x_grid_points": 2,
            "n_y_grid_points": 2,
            "background_averaging": background_averaging,
        }
    )

    _read_rotor_averaging(rotor_averaging, algo_dict, verbosity=0)

    assert algo_dict["partial_wakes"] == expected


def test_rotor_averaging_rejects_non_square_grid():
    rotor_averaging = Dict(
        {
            "name": "area_overlap",
            "grid": "grid",
            "n_x_grid_points": 2,
            "n_y_grid_points": 3,
        }
    )

    with pytest.raises(NotImplementedError, match="Only nx=ny supported"):
        _read_rotor_averaging(rotor_averaging, {}, verbosity=0)
