import matplotlib.pyplot as plt
import numpy as np
import pytest

from foxes.utils.geom2d import AreaIntersection, ClosedPolygon


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
