from __future__ import annotations

from typing import Any

import numpy as np


def plot_wind_rose_bars(
    ax: Any,
    frequency: np.ndarray,
    wd_edges: np.ndarray,
    *,
    cmap: str = "summer",
) -> np.ndarray:
    """Plot stacked wind-rose bars and return the speed-bin colors.

    Parameters
    ----------
    ax
        A Matplotlib polar axes object.
    frequency
        Frequencies with shape ``(n_wd_bins, n_ws_bins)``.
    wd_edges
        Wind-direction bin edges in meteorological degrees.
    cmap
        Matplotlib color-map name.

    Returns
    -------
    numpy.ndarray
        RGBA colors assigned to the wind-speed bins.

    Raises
    ------
    ValueError
        If the frequency array does not have one direction axis entry per
        direction-bin interval.
    """
    frequency = np.asarray(frequency)
    wd_edges = np.asarray(wd_edges)
    if frequency.ndim != 2 or frequency.shape[0] != len(wd_edges) - 1:
        raise ValueError(
            "Wind-rose frequencies must have shape "
            "(n_wd_bins, n_ws_bins)"
        )

    n_ws_bins = frequency.shape[1]
    wd_centers = 0.5 * (wd_edges[:-1] + wd_edges[1:])
    wd_angles = np.radians(np.mod(90.0 - wd_centers, 360.0))
    wd_widths = np.radians(np.diff(wd_edges)) * 0.9
    import matplotlib.pyplot as plt

    colors = plt.get_cmap(cmap, n_ws_bins)(np.linspace(0.0, 1.0, n_ws_bins))
    bottom = np.zeros(frequency.shape[0])
    for ws_i in range(n_ws_bins):
        ax.bar(
            wd_angles,
            frequency[:, ws_i],
            bottom=bottom,
            width=wd_widths,
            color=colors[ws_i],
            align="center",
        )
        bottom += frequency[:, ws_i]

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    return colors
