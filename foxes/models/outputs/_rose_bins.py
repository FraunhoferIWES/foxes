"""Reduce turbine fields into wind bins and render polar heatmaps."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.projections.polar import PolarAxes
from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV


def _bins_get_data(
    results: Dataset,
    output: Any,
    variable: str,
    ws_bins: Any,
    wd_sectors: int = 12,
    wd_var: str = FV.AMB_WD,
    ws_var: str = FV.AMB_REWS,
    turbine: int = 0,
    contraction: str = "weights",
) -> Dataset:
    """Reduce turbine values within wind-direction and wind-speed bins.

    Parameters
    ----------
    results
        Read-only farm fields with dimensions ``(FC.STATE, FC.TURBINE)``.
        ``FV.WEIGHT`` may instead have dimensions ``(FC.STATE,)``.
    output
        Executing output context; not read or mutated by this reduction.
    variable
        The variable name
    ws_bins
        Wind-speed bin edges in m/s.
    wd_sectors
        Number of equal direction sectors, centred on north at zero degrees.
    wd_var
        The wind direction variable
    ws_var
        The wind speed variable
    turbine
        Positional turbine index for fields and turbine-dependent weights.
    contraction
        ``"weights"`` divides weighted values by literal bin-weight sums;
        ``"mean_no_weights"`` divides sums by counts; ``"sum_no_weights"``
        leaves sums unnormalized.

    Returns
    -------
    data
        New dataset with the variable on dimensions ``(wd_var, ws_var)``.
        Coordinates are bin centres; ``*_bounds`` attributes contain edges.
        Weighted bins below ``1e-13`` and empty mean bins become NaN.
        Unweighted empty sums remain zero. Input arrays are not modified.

    Raises
    ------
    ValueError
        If weight dimensions are neither ``(FC.STATE,)`` nor
        ``(FC.STATE, FC.TURBINE)``.
    KeyError
        If ``contraction`` is not supported.
    """
    if results[FV.WEIGHT].dims == (FC.STATE,):
        w = results[FV.WEIGHT].to_numpy()
    elif results[FV.WEIGHT].dims == (FC.STATE, FC.TURBINE):
        w = results[FV.WEIGHT].to_numpy()[:, turbine]
    else:
        raise ValueError(
            f"Wrong dimensions for '{FV.WEIGHT}'. Expecting {(FC.STATE,)} or {(FC.STATE, FC.TURBINE)}, got {results[FV.WEIGHT].dims}"
        )
    var = results[variable].to_numpy()[:, turbine]
    ws = results[ws_var].to_numpy()[:, turbine]
    wd = results[wd_var].to_numpy()[:, turbine].copy()
    wd_delta = 360 / wd_sectors
    wd[wd >= 360 - wd_delta / 2] -= 360
    wd_bins = np.arange(-wd_delta / 2, 360, wd_delta)
    ws_bins = np.asarray(ws_bins)
    if contraction == "weights":
        z = np.histogram2d(wd, ws, (wd_bins, ws_bins), weights=w)[0]
        z[z < 1e-13] = np.nan
        z = np.histogram2d(wd, ws, (wd_bins, ws_bins), weights=w * var)[0] / z
    elif contraction == "mean_no_weights":
        z = np.histogram2d(wd, ws, (wd_bins, ws_bins))[0].astype(w.dtype)
        z[z < 1] = np.nan
        z = np.histogram2d(wd, ws, (wd_bins, ws_bins), weights=var)[0] / z
    elif contraction == "sum_no_weights":
        z = np.histogram2d(wd, ws, (wd_bins, ws_bins), weights=var)[0]
    else:
        raise KeyError(
            f"Contraction '{contraction}' not supported. Choices: weights, mean_no_weights, sum_no_weights"
        )
    data = Dataset(
        coords={
            wd_var: 0.5 * (wd_bins[:-1] + wd_bins[1:]),
            ws_var: 0.5 * (ws_bins[:-1] + ws_bins[1:]),
        },
        data_vars={variable: ((wd_var, ws_var), z)},
        attrs={f"{wd_var}_bounds": wd_bins, f"{ws_var}_bounds": ws_bins},
    )
    return data


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
