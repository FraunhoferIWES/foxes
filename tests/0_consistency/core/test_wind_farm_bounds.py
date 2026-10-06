import numpy as np
import pytest

import foxes


def test_xy_bounds_contain_positions_from_all_states():
    state_xy = np.array([[0.0, 5.0], [10.0, -5.0], [-3.0, 8.0]])
    static_xy = np.array([20.0, 15.0])
    farm = foxes.WindFarm()
    farm.add_turbine(foxes.Turbine(state_xy), verbosity=0)
    farm.add_turbine(foxes.Turbine(static_xy), verbosity=0)

    xy_min, xy_max = farm.get_xy_bounds(extra_space=2.0)

    all_xy = np.concatenate((state_xy, static_xy[None, :]), axis=0)
    assert np.all(all_xy >= xy_min)
    assert np.all(all_xy <= xy_max)
    np.testing.assert_array_equal(xy_min, [-5.0, -7.0])
    np.testing.assert_array_equal(xy_max, [22.0, 17.0])


def test_xy_bounds_reject_invalid_turbine_positions():
    farm = foxes.WindFarm()
    farm.add_turbine(foxes.Turbine(np.zeros((2, 3))), verbosity=0)

    with pytest.raises(ValueError, match="final dimension 2"):
        farm.get_xy_bounds()
