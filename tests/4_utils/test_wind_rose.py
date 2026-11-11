"""Physical bearing orientation of the shared wind-rose renderer."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from foxes.utils import plot_wind_rose_bars


@pytest.mark.parametrize(
    "direction,expected",
    [(0.0, [0.0, 1.0]), (90.0, [1.0, 0.0]), (180.0, [0.0, -1.0]), (270.0, [-1.0, 0.0])],
)
def test_rose_bars_point_to_the_meteorological_compass_bearing(direction, expected):
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    try:
        plot_wind_rose_bars(
            ax, np.array([[50.0]]), np.array([direction - 15.0, direction + 15.0])
        )
        fig.canvas.draw()
        bar = ax.patches[0]
        angle = bar.get_x() + bar.get_width() / 2
        origin = ax.transData.transform((0.0, 0.0))
        endpoint = ax.transData.transform((angle, bar.get_height()))
        vector = endpoint - origin
        np.testing.assert_allclose(
            vector / np.linalg.norm(vector), expected, atol=1e-12
        )
    finally:
        plt.close(fig)


def test_rose_bars_reject_inconsistent_direction_edges():
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    try:
        with pytest.raises(ValueError, match="n_wd_bins"):
            plot_wind_rose_bars(ax, np.ones((2, 1)), np.array([0.0, 30.0]))
    finally:
        plt.close(fig)
