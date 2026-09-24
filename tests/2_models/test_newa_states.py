import numpy as np
import pytest

from foxes.core import MData
from foxes.input.states import NEWAStates
import foxes.variables as FV


def _interpolation_data():
    grid_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    values = np.array([[0.0], [1.0], [1.0]])
    return grid_points, values


def test_newa_interpolates_spatial_data_without_state_labels():
    states = NEWAStates("unused.nc", output_vars=[FV.WS])
    grid_points, values = _interpolation_data()

    result = states.interpolate_data(
        MData(),
        [FV.X, FV.Y],
        values,
        np.array([[0.25, 0.25]]),
        [FV.WS],
        state_labels=None,
        gpts=grid_points,
    )

    np.testing.assert_allclose(result, [[0.5]])


def test_newa_reports_spatial_interpolation_error_without_state_labels():
    states = NEWAStates("unused.nc", output_vars=[FV.WS])
    grid_points, values = _interpolation_data()

    with pytest.raises(ValueError, match="outside of bounds"):
        states.interpolate_data(
            MData(),
            [FV.X, FV.Y],
            values,
            np.array([[2.0, 2.0]]),
            [FV.WS],
            state_labels=None,
            gpts=grid_points,
        )


def test_newa_reports_state_data_error_without_state_labels():
    states = NEWAStates("unused.nc", output_vars=[FV.WS])
    grid_points, values = _interpolation_data()
    state_values = np.repeat(values[:, None, :], 2, axis=1)

    with pytest.raises(ValueError, match="outside of bounds"):
        states.interpolate_data(
            MData(),
            [FV.X, FV.Y],
            state_values,
            np.array([[2.0, 2.0]]),
            [FV.WS],
            state_labels=None,
            gpts=grid_points,
        )
