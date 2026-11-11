"""Read, render and export farm layouts without retaining calculation state."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from xarray import Dataset

import foxes.variables as FV
from foxes.config import config

from ._layout_rendering import (
    _add_turbine_circles as _add_turbine_circles,
    _annotate_turbines,
    _color_values,
    _get_auto_figsize as _get_auto_figsize,
    _get_turbine_diameters as _get_turbine_diameters,
    _layout_axes,
    _layout_diameters,
    _plot_markers,
    _reference_markers,
)

if TYPE_CHECKING:
    from foxes.core import Algorithm, WindFarm


def _get_layout_data(
    farm: WindFarm,
    fres: Dataset | None,
    algo: Algorithm | None,
    output: Any,
    lonlat: bool = False,
) -> np.ndarray:
    """Collect turbine positions in farm order without changing their owners.

    Parameters
    ----------
    farm
        Farm supplying turbine positions and geographic coordinates.
    fres
        Turbine-only ``FV.X``, ``FV.Y`` and ``FV.H`` fields, required when
        ``output.from_results`` is enabled.
    algo
        Algorithm context; not read or mutated.
    output
        Executing output supplying ``from_results``.
    lonlat
        Use farm longitude and latitude instead of projected x and y.

    Returns
    -------
    positions
        New configured-double array of shape ``(n_turbines, 3)``. Columns are
        x, y, hub height in metres, or longitude, latitude in degrees and hub
        height in metres. Geographic mode always uses turbine-instance heights.

    Raises
    ------
    ValueError
        If geographic coordinates are requested but unavailable.
    """
    positions = np.zeros((farm.n_turbines, 3), dtype=config.dtype_double)
    if lonlat:
        if not farm.has_lonlat():
            raise ValueError(
                f"WindFarm '{farm.name}': lonlat coordinates not available"
            )
        positions[:, :2] = farm.lonlat
        positions[:, 2] = [turbine.H for turbine in farm.turbines]
    elif output.from_results:
        assert fres is not None
        for axis, variable in enumerate((FV.X, FV.Y, FV.H)):
            positions[:, axis] = fres[variable].to_numpy()
    else:
        for index, turbine in enumerate(farm.turbines):
            positions[index, :2] = turbine.xy
            positions[index, 2] = turbine.H
    return positions


def _get_layout_dict(
    farm: WindFarm, fres: Dataset | None, algo: Algorithm | None, output: Any
) -> dict[str, dict[str, dict[str, Any]]]:
    """Collect projected turbine positions in the layout JSON schema.

    Parameters
    ----------
    farm
        Farm supplying the top-level name and turbine identities.
    fres
        Turbine-only coordinate fields when results-based layout is enabled.
    algo
        Algorithm context passed to coordinate lookup.
    output
        Executing output supplying ``from_results``.

    Returns
    -------
    layout
        New mapping keyed by farm name, then turbine name or stringified index.
        Records contain ``id``, ``name``, ``UTMX`` and ``UTMY`` in metres;
        hub heights are not included.
    """
    positions = _get_layout_data(farm, fres, algo, output)
    layout: dict[str, dict[str, dict[str, Any]]] = {farm.name: {}}
    for turbine, position in zip(farm.turbines, positions):
        layout[farm.name][turbine.name or str(turbine.index)] = {
            "id": turbine.index,
            "name": turbine.name,
            "UTMX": position[0],
            "UTMY": position[1],
        }
    return layout


def _get_figure(
    farm: WindFarm,
    fres: Dataset | None,
    algo: Algorithm | None,
    output: Any,
    color_by: str | None = None,
    fontsize: int = 8,
    figsize: Any = None,
    annotate: int = 1,
    title: str | None = None,
    fig: Figure | None = None,
    ax: Axes | None = None,
    normalize_D: bool = False,
    ret_im: bool = False,
    bargs: dict[str, Any] | None = None,
    legend_labels: dict[str, str] | None = None,
    anno_delx: float = 0,
    anno_dely: float = 0,
    lonlat: bool = False,
    true_turbine_radii: bool = False,
    **kwargs: Any,
) -> Any:
    """Render a farm layout with existing marker and boundary conventions.

    Parameters
    ----------
    farm
        Read-only farm geometry and turbine ordering.
    fres
        Farm fields for results-based coordinates or color reductions.
    algo
        Initialized algorithm resolving missing rotor diameters.
    output
        Executing output supplying ``nofig``, ``D`` and layout-source options.
    color_by
        Turbine-only field, farm/cluster labels, or a field prefixed by
        ``mean_``, ``sum_``, ``min_`` or ``max_``. Means use literal weights.
    fontsize
        Turbine annotation size in points.
    figsize
        New figure dimensions in inches; ``None`` derives them from extents.
    annotate
        ``0`` omits annotations, ``1`` uses positional turbine indices,
        ``2`` turbine names and ``3`` wind-farm names at their centres.
    title
        Explicit title; ``None`` derives it from the farm unless suppressed
        by wind-farm annotations or automatic reference-point styling.
    fig
        Figure whose first axes is reused when ``ax`` is absent. ``None``
        creates a new figure and subplot, regardless of ``ax``.
    ax
        Caller-owned axes used when ``fig`` is supplied; not cleared or replaced.
    normalize_D
        Divide projected coordinates and physical radii by a uniform diameter.
        Longitude and latitude coordinates are not normalized.
    ret_im
        Include the marker collection with the returned axes.
    bargs
        Copied boundary options. ``boundary`` overrides the plotted geometry;
        ``None`` suppresses it without changing farm bounds.
    legend_labels
        Color-to-label mapping for an upper-left legend; not modified.
        Named reference-point turbines receive red markers and automatic labels
        unless explicit colors or field coloring are supplied.
    anno_delx
        Turbine-label horizontal offset in plotted coordinate units.
    anno_dely
        Turbine-label vertical offset in plotted coordinate units.
    lonlat
        Use geographic coordinates in degrees instead of projected metres.
    true_turbine_radii
        Draw physical rotor circles instead of scatter markers.
    kwargs
        Scatter or patch-collection options, copied before applying defaults.

    Returns
    -------
    plot
        Updated axes, or ``(ax, collection)`` when ``ret_im`` is enabled.
        Categorical coloring returns its final collection. Disabled figures
        return ``(None, None)`` irrespective of ``ret_im``. No figure is saved,
        shown or closed.

    Raises
    ------
    ValueError
        If physical radii are combined with geographic coordinates, required
        diameters are invalid, normalization needs nonuniform diameters,
        coloring lacks results, or weight dimensions are unsupported.
    KeyError
        If ``color_by`` is unknown.
    """
    if output.nofig:
        return None, None
    if true_turbine_radii and lonlat:
        raise ValueError("True turbine radii are not available for lon/lat plots")
    data = _get_layout_data(farm, fres, algo, output, lonlat=lonlat)
    diameter, diameters = _layout_diameters(
        farm, fres, algo, output, normalize_D, true_turbine_radii
    )
    min_diameter = (
        float(np.min(diameters))
        if true_turbine_radii and diameters is not None
        else None
    )
    fig, ax = _layout_axes(
        farm, fres, algo, output, data, lonlat, min_diameter, figsize, fig, ax
    )
    x = None
    if farm.n_turbines:
        x = data[:, 0] / diameter if normalize_D and not lonlat else data[:, 0]
        y = data[:, 1] / diameter if normalize_D and not lonlat else data[:, 1]
        options, title, legend_labels = _reference_markers(
            farm, color_by, kwargs, title, legend_labels
        )
        if color_by is not None:
            options["c"] = _color_values(farm, fres, color_by)
        colors = options.pop("c", "orange")
        radii = None
        if true_turbine_radii:
            assert diameters is not None
            radii = diameters / (2.0 * diameter) if normalize_D else diameters / 2.0
        image, categorical = _plot_markers(
            farm, fres, algo, output, ax, x, y, radii, colors, color_by, options
        )
        _annotate_turbines(farm, ax, x, y, annotate, fontsize, anno_delx, anno_dely)
    _decorate_layout(farm, ax, bargs, title, annotate, diameter, normalize_D, lonlat)
    if x is None or (min(x) != max(x) and min(y) != max(y)):
        ax.set_aspect("equal", adjustable="box")
    ax.autoscale_view(tight=True)
    if color_by is not None and not categorical:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(image, cax=cax)
    _add_layout_legend(ax, legend_labels)
    return (ax, image) if ret_im else ax


def _decorate_layout(
    farm: WindFarm,
    ax: Axes,
    bargs: dict | None,
    title: str | None,
    annotate: int,
    diameter: float | None,
    normalize_D: bool,
    lonlat: bool,
) -> None:
    """Add the configured boundary, title and coordinate labels to layout axes.

    Parameters
    ----------
    farm
        Farm providing default boundary and title.
    ax
        Axes to update in place.
    bargs
        Boundary override and plotting options; not modified.
    title
        Explicit title, or ``None`` for automatic farm naming.
    annotate
        Annotation mode; mode three suppresses an automatic title.
    diameter
        Diameter in metres included in normalized automatic titles.
    normalize_D
        Label projected coordinates in rotor-diameter units.
    lonlat
        Label coordinates as longitude and latitude in degrees.
    """
    options: dict[str, Any] = {"fill_mode": "inside_lightgray"}
    if bargs is not None:
        options.update(bargs)
    boundary = options.pop("boundary", farm.boundary)
    if boundary is not None:
        boundary.add_to_figure(ax, **options)
    if title is not None or annotate != 3:
        automatic = (
            farm.name
            if diameter is None or not normalize_D
            else f"{farm.name} (D = {diameter} m)"
        )
        ax.set_title(title if title is not None else automatic)
    ax.set_xlabel("Longitude [deg]" if lonlat else "x [D]" if normalize_D else "x [m]")
    ax.set_ylabel("Latitude [deg]" if lonlat else "y [D]" if normalize_D else "y [m]")
    ax.grid()


def _add_layout_legend(ax: Axes, legend_labels: dict[str, str] | None) -> None:
    """Extend existing legend entries with explicit marker-color labels.

    Parameters
    ----------
    ax
        Axes receiving an upper-left legend.
    legend_labels
        Read-only color-to-label mapping; empty or ``None`` adds no legend.
    """
    if not legend_labels:
        return
    handles, labels = ax.get_legend_handles_labels()
    entries = dict(zip(labels, handles))
    for color, label in legend_labels.items():
        entries.setdefault(
            label,
            Line2D([], [], color=color, marker="o", linestyle="none", label=label),
        )
    ax.legend(list(entries.values()), list(entries), loc="upper left")


def _write_plot(
    farm: WindFarm,
    fres: Dataset | None,
    algo: Algorithm | None,
    output: Any,
    file_name: str | None = None,
    fontsize: int = 8,
    **kwargs: Any,
) -> None:
    """Save a layout figure, then optionally display it, and always close it.

    Parameters
    ----------
    farm
        Farm to render and source of the default ``<farm.name>.png`` name.
    fres
        Farm fields used by results-based coordinates and coloring.
    algo
        Algorithm supplying missing turbine geometry.
    output
        Executing output supplying path resolution, ``nofig`` and show control.
    file_name
        File name resolved through ``output.get_fpath``; ``None`` uses the farm.
    fontsize
        Turbine annotation size in points.
    kwargs
        Options forwarded to ``_get_figure``.

    Notes
    -----
    Disabled figures perform no I/O. Saving precedes display. The figure is
    closed even when saving or display fails, including caller-supplied figures.
    """
    if output.nofig:
        return
    ax = _get_figure(
        farm, fres, algo, output, fontsize=fontsize, ret_im=False, **kwargs
    )
    fig = ax.get_figure()
    name = file_name if file_name is not None else farm.name + ".png"
    path = output.get_fpath(name)
    try:
        fig.savefig(path, bbox_inches="tight")
        output._show_figures()
    finally:
        plt.close(fig)


def _write_xyh(
    farm: WindFarm,
    fres: Dataset | None,
    algo: Algorithm | None,
    output: Any,
    file_path: str | None = None,
) -> None:
    """Write turbine coordinates as whitespace-delimited layout rows.

    Parameters
    ----------
    farm
        Farm supplying geometry and the default ``<farm.name>.xyh`` name.
    fres
        Turbine-only coordinate results when results-based layout is enabled.
    algo
        Algorithm context passed to coordinate lookup.
    output
        Executing output supplying layout-source options.
    file_path
        Direct filesystem path, not resolved through the output directory.
        ``None`` writes the farm-named file in the current directory.

    Notes
    -----
    Rows contain x, y and hub height in metres. Geographic farms prepend
    longitude, latitude and geographic-mode hub height, while retaining the
    established ``lon lat x y h`` header.
    """
    name = file_path if file_path is not None else farm.name + ".xyh"
    data = _get_layout_data(farm, fres, algo, output, lonlat=False)
    if not farm.has_lonlat():
        np.savetxt(name, data, header="x y h")
    else:
        data = np.concatenate(
            (_get_layout_data(farm, fres, algo, output, lonlat=True), data), axis=1
        )
        np.savetxt(name, data, header="lon lat x y h")


def _get_dataframe(
    farm: WindFarm,
    fres: Dataset | None,
    algo: Algorithm | None,
    output: Any,
    type_col: str | None = None,
    type_algo: Algorithm | None = None,
    col_farm: str = "wind_farm",
    col_cluster: str = "cluster",
) -> pd.DataFrame:
    """Collect rounded layout columns in farm turbine order.

    Parameters
    ----------
    farm
        Farm supplying names, geometry and group membership.
    fres
        Turbine-only coordinates when results-based layout is enabled.
    algo
        Initialized algorithm resolving missing heights and rotor diameters.
    output
        Executing output supplying layout-source options.
    type_col
        Turbine-type column name, or ``None`` to omit it.
    type_algo
        Initialized algorithm supplying turbine types; ``None`` uses ``algo``.
    col_farm
        Wind-farm membership column name, included only for multiple wind farms.
    col_cluster
        Cluster membership column name, included only for multiple clusters.

    Returns
    -------
    layout
        New DataFrame indexed by positional ``index``. Columns include name,
        x/y in metres (four decimals), hub height and diameter in metres (two
        decimals), and available lon/lat in degrees (six decimals), followed
        by group columns and a requested type column.

    Raises
    ------
    ValueError
        If turbine-type export has no initialized algorithm with turbine types.
    """
    lonlat = farm.has_lonlat()
    columns = (
        ["name", "lon", "lat", "x", "y", "h", "D"]
        if lonlat
        else ["name", "x", "y", "h", "D"]
    )
    if farm.wind_farm_names is not None and len(farm.wind_farm_names) > 1:
        columns.append(col_farm)
    if farm.cluster_names is not None and len(farm.cluster_names) > 1:
        columns.append(col_cluster)
    layout = pd.DataFrame(index=range(farm.n_turbines), columns=columns)
    layout.index.name = "index"
    layout["name"] = [turbine.name for turbine in farm.turbines]
    if lonlat:
        data = _get_layout_data(farm, fres, algo, output, lonlat=True)
        layout["lon"] = np.round(data[:, 0], 6)
        layout["lat"] = np.round(data[:, 1], 6)
    data = _get_layout_data(farm, fres, algo, output, lonlat=False)
    layout["x"], layout["y"], layout["h"] = (
        np.round(data[:, 0], 4),
        np.round(data[:, 1], 4),
        np.round(data[:, 2], 2),
    )
    _fill_turbine_geometry(farm, algo, layout)
    if type_col is not None:
        type_algo = algo if type_algo is None else type_algo
        if type_algo is None or type_algo.farm_controller.turbine_types is None:
            raise ValueError("Turbine type export requires an initialized algorithm")
        layout[type_col] = [
            model.name for model in type_algo.farm_controller.turbine_types
        ]
    if col_farm in columns:
        layout[col_farm] = farm.wind_farm_list
    if col_cluster in columns:
        layout[col_cluster] = farm.cluster_list
    return layout


def _fill_turbine_geometry(
    farm: WindFarm, algo: Algorithm | None, layout: pd.DataFrame
) -> None:
    """Fill unresolved hub heights and rotor diameters in a layout table.

    Parameters
    ----------
    farm
        Farm supplying turbine-instance or model-resolved geometry.
    algo
        Initialized algorithm for geometry lookup, or ``None`` for instances.
    layout
        Table modified in place. If any height is NaN, replace all heights;
        rotor diameters are always populated. Model values use two decimals.
    """
    if np.any(np.isnan(layout["h"])):
        layout["h"] = (
            np.round(farm.get_hub_heights(algo=algo), 2)
            if algo is not None
            else [turbine.H for turbine in farm.turbines]
        )
    layout["D"] = (
        np.round(farm.get_rotor_diameters(algo=algo), 2)
        if algo is not None
        else [turbine.D for turbine in farm.turbines]
    )


def _write_csv(
    farm: WindFarm,
    fres: Dataset | None,
    algo: Algorithm | None,
    output: Any,
    file_name: str | None = None,
    verbosity: int = 1,
    **kwargs: Any,
) -> None:
    """Write the rounded layout DataFrame to a resolved CSV path.

    Parameters
    ----------
    farm
        Farm to export; supplies the default ``<farm.name>.csv`` name.
    fres
        Turbine-only coordinate results when results-based layout is enabled.
    algo
        Initialized algorithm for geometry or turbine-type lookup.
    output
        Executing output resolving the file path.
    file_name
        Output file name, or ``None`` for the farm-named file.
    verbosity
        Print the resolved path above zero.
    kwargs
        Options forwarded to ``_get_dataframe``, not to pandas CSV writing.
    """
    name = file_name if file_name is not None else farm.name + ".csv"
    path = output.get_fpath(name)
    if verbosity > 0:
        print(f"Writing farm layout to '{path}'")
    _get_dataframe(farm, fres, algo, output, **kwargs).to_csv(path)


def _write_json(
    farm: WindFarm,
    fres: Dataset | None,
    algo: Algorithm | None,
    output: Any,
    file_name: str | None = None,
) -> None:
    """Write named turbine identities and projected coordinates as layout JSON.

    Parameters
    ----------
    farm
        Farm to export; supplies the default ``<farm.name>.json`` name.
    fres
        Turbine-only coordinate results when results-based layout is enabled.
    algo
        Algorithm context passed to coordinate lookup.
    output
        Executing output resolving the file path and layout source.
    file_name
        Output file name, or ``None`` for the farm-named file.
    """
    data = _get_layout_dict(farm, fres, algo, output)
    name = file_name if file_name is not None else farm.name + ".json"
    with open(output.get_fpath(name), "w") as stream:
        json.dump(data, stream, indent=4)
