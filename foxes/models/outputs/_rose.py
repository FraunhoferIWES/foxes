"""Calculate literal-weight wind-rose frequencies and render polar bars."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.projections.polar import PolarAxes
from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV
from foxes.utils import plot_wind_rose_bars


def _rose_get_data(
    results: Dataset,
    output: Any,
    wd_sectors: int,
    ws_var: str,
    ws_bins: Any,
    wd_var: str = FV.AMB_WD,
    turbine: int = 0,
    point: int = 0,
    add_inf: bool = False,
) -> Dataset:
    """Bin literal state weights into wind-direction and wind-speed sectors.

    Parameters
    ----------
    results
        Read-only farm or point results. Wind fields have dimensions
        ``(FC.STATE, FC.TURBINE)`` or ``(FC.STATE, FC.POINT)``.
    output
        Executing output; ``result_dimension`` selects turbines or points.
    wd_sectors
        Number of equal direction sectors, centred on north at zero degrees.
    ws_var
        Wind-speed field name, with values in m/s.
    ws_bins
        Wind-speed bin edges in m/s, including the final right edge.
    wd_var
        Wind-direction field name, with values in degrees clockwise from north.
    turbine
        Positional turbine index for turbine-dependent weights and farm fields.
    point
        Positional point index for point-dependent weights and point fields.
    add_inf
        Append positive infinity as a final wind-speed edge.

    Returns
    -------
    data
        New dataset with dimensions ``(wd_var, ws_var)`` and ``frequency`` in
        percent. Coordinates are bin centres; ``bin_min_*`` and ``bin_max_*``
        store edges and ``*_bounds`` attributes store complete edge arrays.
        State weights are multiplied by 100 without normalization.

    Raises
    ------
    ValueError
        If weights do not have dimensions ``(FC.STATE,)``,
        ``(FC.STATE, FC.TURBINE)`` or ``(FC.STATE, FC.POINT)``.
    """
    assert results is not None
    if results[FV.WEIGHT].dims == (FC.STATE,):
        w = results[FV.WEIGHT].to_numpy()
    elif results[FV.WEIGHT].dims == (FC.STATE, FC.TURBINE):
        w = results[FV.WEIGHT].to_numpy()[:, turbine]
    elif results[FV.WEIGHT].dims == (FC.STATE, FC.POINT):
        w = results[FV.WEIGHT].to_numpy()[:, point]
    else:
        raise ValueError(
            f"Wrong dimensions for '{FV.WEIGHT}'. Expecting {(FC.STATE,)}, {(FC.STATE, FC.TURBINE)} or {(FC.STATE, FC.POINT)}, got {results[FV.WEIGHT].dims}"
        )
    if add_inf:
        ws_bins = list(ws_bins) + [np.inf]
    t = turbine if output.result_dimension == FC.TURBINE else point
    ws = results[ws_var].to_numpy()[:, t]
    wd = results[wd_var].to_numpy()[:, t].copy()
    wd_delta = 360 / wd_sectors
    wd[wd >= 360 - wd_delta / 2] -= 360
    wd_bins = np.arange(-wd_delta / 2, 360, wd_delta)
    ws_bins = np.asarray(ws_bins, dtype=ws.dtype)
    freq = 100 * np.histogram2d(wd, ws, (wd_bins, ws_bins), weights=w)[0]
    data = Dataset(
        coords={
            wd_var: np.arange(0, 360, wd_delta),
            ws_var: 0.5 * (ws_bins[:-1] + ws_bins[1:]),
        },
        data_vars={
            f"bin_min_{wd_var}": (wd_var, wd_bins[:-1]),
            f"bin_max_{wd_var}": (wd_var, wd_bins[1:]),
            f"bin_min_{ws_var}": (ws_var, ws_bins[:-1]),
            f"bin_max_{ws_var}": (ws_var, ws_bins[1:]),
            "frequency": ((wd_var, ws_var), freq),
        },
        attrs={f"{wd_var}_bounds": wd_bins, f"{ws_var}_bounds": ws_bins},
    )
    return data


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
