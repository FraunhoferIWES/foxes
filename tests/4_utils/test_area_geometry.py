from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pytest

from foxes.utils.geom2d import (
    AreaGeometry,
    AreaIntersection,
    Circle,
    ClosedPolygon,
    HalfPlane,
)


def make_intersection() -> AreaIntersection:
    left = ClosedPolygon(np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]))
    right = ClosedPolygon(np.array([[1.0, 0.0], [3.0, 0.0], [3.0, 2.0], [1.0, 2.0]]))
    return AreaIntersection([left, right])


def test_area_intersection_boundary() -> None:
    fig, ax = plt.subplots()
    try:
        make_intersection().add_to_figure(
            ax,
            True,
            None,
            pars_boundary={"edgecolor": "red", "zorder": 12},
        )

        assert len(ax.patches) == 2
        assert all(patch.get_zorder() == 12 for patch in ax.patches)
        assert all(
            patch.get_edgecolor() == (1.0, 0.0, 0.0, 1.0) for patch in ax.patches
        )
    finally:
        plt.close(fig)


def test_area_intersection_without_overlay() -> None:
    fig, ax = plt.subplots()
    try:
        make_intersection().add_to_figure(
            ax,
            show_boundary=False,
            fill_mode=None,
        )

        assert not ax.patches
        assert not ax.collections
    finally:
        plt.close(fig)


def test_area_intersection_invalid_fill_mode() -> None:
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="Illegal parameter"):
            make_intersection().add_to_figure(ax, fill_mode="invalid")
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "geometry, points, expected_inside, expected_distances, expected_nearest",
    [
        pytest.param(
            Circle(np.array([0.0, 0.0]), 2.0),
            np.array([[1.0, 0.0], [3.0, 0.0], [0.0, -0.5]]),
            np.array([True, False, True]),
            np.array([1.0, 1.0, 1.5]),
            np.array([[2.0, 0.0], [2.0, 0.0], [0.0, -2.0]]),
            id="circle",
        ),
        pytest.param(
            HalfPlane([0.0, 0.0], [1.0, 0.0]),
            np.array([[-2.0, 1.0], [0.0, 1.0], [3.0, -1.0]]),
            np.array([False, True, True]),
            np.array([2.0, 0.0, 3.0]),
            np.array([[0.0, 1.0], [0.0, 1.0], [0.0, -1.0]]),
            id="half-plane",
        ),
        pytest.param(
            ClosedPolygon(
                np.array([[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]])
            ),
            np.array([[0.0, 0.5], [3.0, 0.0], [-1.5, 0.0]]),
            np.array([True, False, True]),
            np.array([0.5, 1.0, 0.5]),
            np.array([[0.0, 1.0], [2.0, 0.0], [-2.0, 0.0]]),
            id="polygon",
        ),
    ],
)
def test_primitive_geometry_queries(
    geometry: AreaGeometry,
    points: np.ndarray,
    expected_inside: np.ndarray,
    expected_distances: np.ndarray,
    expected_nearest: np.ndarray,
) -> None:
    distances, nearest = geometry.points_distance(points, return_nearest=True)

    np.testing.assert_array_equal(geometry.points_inside(points), expected_inside)
    np.testing.assert_allclose(distances, expected_distances)
    np.testing.assert_allclose(nearest, expected_nearest)


def test_circle_centre_returns_boundary_nearest_point() -> None:
    circle = Circle(np.array([1.0, 2.0]), 3.0)
    points = np.array([[1.0, 2.0], [1.0, 3.0], [3.0, 2.0]])

    distances, nearest = circle.points_distance(points, return_nearest=True)

    np.testing.assert_allclose(distances, [3.0, 2.0, 1.0])
    np.testing.assert_allclose(circle.points_distance(nearest), 0.0)


@pytest.mark.parametrize(
    "combine, expected_inside",
    [
        pytest.param(
            lambda left, middle, right, far: left + middle,
            [True, True, False, False, False],
            id="binary",
        ),
        pytest.param(
            lambda left, middle, right, far: left + [middle, right],
            [True, True, True, False, False],
            id="geometry-plus-list",
        ),
        pytest.param(
            lambda left, middle, right, far: (left + middle) + right,
            [True, True, True, False, False],
            id="union-plus-geometry",
        ),
        pytest.param(
            lambda left, middle, right, far: (left + middle) + [right, far],
            [True, True, True, True, False],
            id="union-plus-list",
        ),
        pytest.param(
            lambda left, middle, right, far: left + (middle + right),
            [True, True, True, False, False],
            id="geometry-plus-union",
        ),
        pytest.param(
            lambda left, middle, right, far: (left + middle) + (right + far),
            [True, True, True, True, False],
            id="union-plus-union",
        ),
    ],
)
def test_addition_operator_variants(
    combine: Callable[
        [AreaGeometry, AreaGeometry, AreaGeometry, AreaGeometry], AreaGeometry
    ],
    expected_inside: list[bool],
) -> None:
    areas = [Circle(np.array([x, 0.0]), 1.0) for x in (-6.0, -2.0, 2.0, 6.0)]
    points = np.array([[-6.0, 0.0], [-2.0, 0.0], [2.0, 0.0], [6.0, 0.0], [10.0, 0.0]])

    geometry = combine(*areas)

    np.testing.assert_array_equal(geometry.points_inside(points), expected_inside)


@pytest.mark.parametrize(
    "combine, expected_inside",
    [
        pytest.param(
            lambda outer, left, middle, right, remote: outer - left,
            [False, True, True, True, True, False, False],
            id="single",
        ),
        pytest.param(
            lambda outer, left, middle, right, remote: outer - [left, middle],
            [False, False, True, True, True, False, False],
            id="list",
        ),
        pytest.param(
            lambda outer, left, middle, right, remote: (outer - left) - middle,
            [False, False, True, True, True, False, False],
            id="chained",
        ),
        pytest.param(
            lambda outer, left, middle, right, remote: outer - (left + middle),
            [False, False, True, True, True, False, False],
            id="subtract-union",
        ),
        pytest.param(
            lambda outer, left, middle, right, remote: (outer + remote) - left,
            [False, True, True, True, True, False, True],
            id="subtract-from-union",
        ),
        pytest.param(
            lambda outer, left, middle, right, remote: (outer - left) + left,
            [True, True, True, True, True, False, False],
            id="union-after-subtraction",
        ),
        pytest.param(
            lambda outer, left, middle, right, remote: outer - left.inverse(),
            [True, False, False, False, False, False, False],
            id="intersection-via-subtraction",
        ),
    ],
)
def test_subtraction_operator_variants(
    combine: Callable[
        [AreaGeometry, AreaGeometry, AreaGeometry, AreaGeometry, AreaGeometry],
        AreaGeometry,
    ],
    expected_inside: list[bool],
) -> None:
    outer = Circle(np.array([0.0, 0.0]), 10.0)
    holes = [Circle(np.array([x, 0.0]), 1.0) for x in (-6.0, -2.0, 2.0)]
    remote = Circle(np.array([15.0, 0.0]), 1.0)
    points = np.array(
        [
            [-6.0, 0.0],
            [-2.0, 0.0],
            [2.0, 0.0],
            [6.0, 0.0],
            [0.0, 6.0],
            [12.0, 0.0],
            [15.0, 0.0],
        ]
    )

    geometry = combine(outer, *holes, remote)

    np.testing.assert_array_equal(geometry.points_inside(points), expected_inside)


def test_nested_exclusion_keeps_internal_island() -> None:
    outer = Circle(np.array([0.0, 0.0]), 5.0)
    excluded = Circle(np.array([0.0, 0.0]), 3.0)
    island = Circle(np.array([0.0, 0.0]), 1.0)
    geometry = outer - (excluded - island)
    points = np.array([[0.2, 0.0], [1.5, 0.0], [3.5, 0.0], [6.0, 0.0]])

    distances, nearest = geometry.points_distance(points, return_nearest=True)

    np.testing.assert_array_equal(
        geometry.points_inside(points), [True, False, True, False]
    )
    np.testing.assert_allclose(distances, [0.8, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(
        nearest, [[1.0, 0.0], [1.0, 0.0], [3.0, 0.0], [5.0, 0.0]]
    )


def test_inverse_preserves_distance_and_double_inverse() -> None:
    circle = Circle(np.array([0.0, 0.0]), 2.0)
    inverse = circle.inverse()
    points = np.array([[0.0, 0.0], [3.0, 0.0]])

    np.testing.assert_array_equal(inverse.points_inside(points), [False, True])
    np.testing.assert_allclose(inverse.points_distance(points), [2.0, 1.0])
    assert inverse.inverse() is circle


def test_union_equidistant_nearest_boundary() -> None:
    geometry = Circle(np.array([-2.0, 0.0]), 1.0) + Circle(np.array([2.0, 0.0]), 1.0)

    distances, nearest = geometry.points_distance(
        np.array([[0.0, 0.0]]), return_nearest=True
    )

    np.testing.assert_allclose(distances, [1.0])
    assert any(
        np.allclose(nearest[0], candidate) for candidate in ([-1.0, 0.0], [1.0, 0.0])
    )
