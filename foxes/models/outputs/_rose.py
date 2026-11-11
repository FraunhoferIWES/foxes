"""Render frequency roses and binned turbine fields on polar axes."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.projections.polar import PolarAxes
from xarray import Dataset

import foxes.variables as FV
from foxes.utils import plot_wind_rose_bars


def _rose_get_figure(
    data: Dataset,
    output: Any,
    ws_var: str,
    wd_var: str = FV.AMB_WD,
    fig: Figure | None = None,
    ax: Axes | None = None,
    figsize: Any = None,
    freq_delta: float = 3,
    cmap: str = "summer",
    title: str | None = None,
    legend_pars: dict[str, Any] | None = None,
    ret_data: bool = False,
) -> Any:
    """Draw wind-rose bars on new or caller-owned polar axes.

    Parameters
    ----------
    data
        Read-only frequency dataset on ``(wd_var, ws_var)`` in percent, with
        complete direction and speed edges in ``*_bounds`` attributes.
    output
        Executing output supplying ``nofig``.
    ws_var
        Wind-speed dimension and legend quantity name.
    wd_var
        Wind-direction dimension name.
    fig
        Accepted figure context; axes creation is controlled by ``ax``.
    ax
        Caller-owned polar axes to update without clearing, or ``None`` to
        create a new polar subplot.
    figsize
        Figure dimensions in inches when creating new axes.
    freq_delta
        Radial tick spacing in percent, truncated to an integer.
    cmap
        Matplotlib colormap for wind-speed bins.
    title
        Polar axes title; ``None`` leaves it empty.
    legend_pars
        Matplotlib legend overrides, copied without changing the mapping.
    ret_data
        Include the supplied frequency dataset with the returned axes.

    Returns
    -------
    plot
        Updated polar axes, or ``(ax, data)`` when ``ret_data`` is enabled.
        The returned dataset is the supplied object and is not modified.
        Return ``None`` when ``output.nofig`` is enabled. No save, show or
        close operation is performed.

    Raises
    ------
    TypeError
        If supplied axes are not polar.
    """
    if output.nofig:
        return None
    n_wdb = data.sizes[wd_var]
    ws_bins = np.asarray(data.attrs[f"{ws_var}_bounds"])
    wd_delta = 360 / n_wdb
    freq = data["frequency"].to_numpy()
    if ax is not None:
        if not isinstance(ax, PolarAxes):
            raise TypeError(
                f"Require axes of type '{PolarAxes.__name__}' for '{type(output).__name__}', got '{type(ax).__name__}'"
            )
    else:
        (_fig, ax) = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    color_list = plot_wind_rose_bars(
        ax, freq, data.attrs[f"{wd_var}_bounds"], cmap=cmap
    )
    fmax = np.max(np.sum(freq, axis=1))
    freq_delta = int(freq_delta)
    freq_ticks = np.arange(0, fmax + freq_delta / 2, freq_delta, dtype=np.int32)[1:]
    tksl = np.arange(0, 360, max(wd_delta, 30))
    tks = np.radians(np.mod(90 - tksl, 360))
    ax.set_xticks(tks, [f"{int(d)}°" for d in tksl])
    ax.set_yticks(freq_ticks, [f"{f}%" for f in freq_ticks])
    ax.set_title(title)
    _rose_legend(ax, ws_var, data.sizes[ws_var], ws_bins, color_list, legend_pars)
    return (ax, data) if ret_data else ax


def _rose_legend(
    ax: Axes,
    ws_var: str,
    n_bins: int,
    ws_bins: np.ndarray,
    colors: Any,
    legend_pars: dict[str, Any] | None,
) -> None:
    """Label wind-speed bins in descending order using their plotted colors.

    Parameters
    ----------
    ax
        Polar axes receiving the legend.
    ws_var
        Wind-speed variable name; known speed fields receive m/s units.
    n_bins
        Number of wind-speed bins in the frequency dataset.
    ws_bins
        Read-only wind-speed edges in m/s, shape ``(n_bins + 1,)``.
    colors
        Bin colors returned by ``plot_wind_rose_bars``.
    legend_pars
        Matplotlib legend overrides; not modified.
    """
    handles = [
        Line2D([0], [0], color=color, lw=10) for color in np.flip(colors, axis=0)
    ]
    labels = [
        f"[{ws_bins[i]:.1f}, {ws_bins[i + 1]:.1f})" for i in range(n_bins - 1, -1, -1)
    ]
    options: dict[str, Any] = dict(
        loc="upper left", bbox_to_anchor=(0.8, 0.5), title=f"{ws_var}"
    )
    speed_variables = [FV.WS, FV.REWS, FV.REWS2, FV.REWS3]
    speed_variables += [FV.var2amb[var] for var in speed_variables]
    if ws_var in speed_variables:
        options["title"] += " [m/s]"
    if legend_pars is not None:
        options.update(legend_pars)
    ax.legend(handles, labels, **options)


def _bins_get_figure(
    data: Dataset,
    output: Any,
    variable: str,
    wd_var: str = FV.AMB_WD,
    ws_var: str = FV.AMB_REWS,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    figsize: Any = None,
    ret_data: bool = False,
    **kwargs: Any,
) -> Any:
    """Draw binned turbine values on new or caller-owned polar axes.

    Parameters
    ----------
    data
        Read-only binned field on ``(wd_var, ws_var)`` with complete direction
        and speed edges in ``*_bounds`` attributes.
    output
        Executing output supplying ``nofig``.
    variable
        Binned field name and color-bar title.
    wd_var
        Wind-direction dimension name.
    ws_var
        Wind-speed dimension name.
    fig
        Accepted figure context; axes creation is controlled by ``ax``.
    ax
        Caller-owned polar axes to update without clearing, or ``None`` for
        a new polar subplot.
    title
        Polar axes title; ``None`` leaves it empty.
    figsize
        Figure dimensions in inches when creating new axes.
    ret_data
        Include the unchanged supplied dataset with the returned axes.
    kwargs
        Matplotlib pseudocolor-mesh overrides, copied before applying defaults.

    Returns
    -------
    plot
        Updated axes, or ``(ax, data)`` when ``ret_data`` is enabled.
        Return ``None`` when ``output.nofig`` is enabled. A color bar is added
        to the axes' figure; no figure is saved, shown or closed.

    Raises
    ------
    TypeError
        If supplied axes are not polar.
    """
    if output.nofig:
        return None
    wd_delta = 360 / data.sizes[wd_var]
    wd_bins = np.mod(90 - data.attrs[f"{wd_var}_bounds"], 360)
    wd_bins = np.radians(wd_bins)
    ws_bins = data.attrs[f"{ws_var}_bounds"]
    if ax is not None:
        if not isinstance(ax, PolarAxes):
            raise TypeError(
                f"Require axes of type '{PolarAxes.__name__}' for '{type(output).__name__}', got '{type(ax).__name__}'"
            )
    else:
        (fig, ax) = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    (y, x) = np.meshgrid(ws_bins, wd_bins)
    z = data[variable].to_numpy()
    options = {"shading": "flat"}
    options.update(kwargs)
    img = ax.pcolormesh(x, y, z, **options)
    tksl = np.arange(0, 360, max(wd_delta, 30))
    tks = np.radians(np.mod(90 - tksl, 360))
    ax.set_xticks(tks, [f"{d}°" for d in tksl])
    ax.set_yticks(ws_bins)
    ax.set_title(title)
    cbar = ax.figure.colorbar(img, ax=ax, pad=0.12)
    cbar.ax.set_title(variable)
    return (ax, data) if ret_data else ax
