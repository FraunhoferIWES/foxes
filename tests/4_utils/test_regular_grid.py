import numpy as np
import pytest

from foxes.utils import (
    detect_regular_grid,
    regular_grid_from_points,
    select_grid_axes,
)


def _irregular_perimeter():
    return np.array(
        [
            [x, y, 100.0]
            for x, y in [
                (-200.0, -200.0),
                (-100.0, -200.0),
                (0.0, -200.0),
                (100.0, -200.0),
                (200.0, -200.0),
                (200.0, -100.0),
                (200.0, 0.0),
                (200.0, 100.0),
                (200.0, 200.0),
                (100.0, 200.0),
                (0.0, 200.0),
                (-100.0, 200.0),
                (-200.0, 200.0),
                (-200.0, 100.0),
                (-200.0, 0.0),
                (-200.0, -100.0),
            ]
        ]
    )


def test_detect_regular_grid_accepts_complete_grid_and_duplicates():
    points = np.array(
        [[x, y, h] for x in [0.0, 50.0] for y in [10.0, 20.0] for h in [80.0]]
    )
    points = np.concatenate([points, points[:1]])

    axes = detect_regular_grid(points)

    assert axes is not None
    np.testing.assert_array_equal(axes[0], [0.0, 50.0])
    np.testing.assert_array_equal(axes[1], [10.0, 20.0])
    np.testing.assert_array_equal(axes[2], [80.0])


def test_detect_regular_grid_rejects_missing_axis_combination():
    points = np.array([[0.0, 0.0, 100.0], [0.0, 1.0, 100.0], [1.0, 0.0, 100.0]])

    assert detect_regular_grid(points) is None


def test_select_grid_axes_adds_native_padding():
    axes = select_grid_axes(
        (
            np.arange(-100.0, 201.0, 50.0),
            np.arange(100.0, 351.0, 50.0),
            [80.0, 100.0],
        ),
        bounds=(np.array([0.0, 200.0]), np.array([100.0, 250.0])),
        padding=1,
    )

    np.testing.assert_array_equal(axes[0], [-50.0, 0.0, 50.0, 100.0, 150.0])
    np.testing.assert_array_equal(axes[1], [150.0, 200.0, 250.0, 300.0])
    np.testing.assert_array_equal(axes[2], [80.0, 100.0])


def test_regular_grid_from_irregular_points_infers_resolution():
    axes, source_is_grid = regular_grid_from_points(
        _irregular_perimeter(),
        bounds=(np.array([-50.0, -50.0]), np.array([50.0, 50.0])),
        padding=1,
    )

    assert not source_is_grid
    expected = [-200.0, -100.0, 0.0, 100.0, 200.0]
    np.testing.assert_array_equal(axes[0], expected)
    np.testing.assert_array_equal(axes[1], expected)
    np.testing.assert_array_equal(axes[2], [100.0])


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: detect_regular_grid([[0.0, 1.0]]), "shape"),
        (
            lambda: select_grid_axes(([1.0, 0.0], [0.0, 1.0])),
            "strictly increasing",
        ),
        (
            lambda: select_grid_axes(([0.0, 1.0], [0.0, 1.0]), padding=-1),
            "non-negative",
        ),
        (
            lambda: regular_grid_from_points([[0.0, 0.0, 100.0], [1.0, 1.0, 100.0]]),
            "three horizontal",
        ),
    ],
)
def test_regular_grid_utilities_reject_invalid_input(call, message):
    with pytest.raises(ValueError, match=message):
        call()
