"""Handle layout marker geometry, color reductions and turbine annotations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV
from foxes.config import config

if TYPE_CHECKING:
    from foxes.core import Algorithm, WindFarm

_MIN_ROTOR_DIAMETER_PIXELS = 4.0


def _get_auto_figsize(
    farm: WindFarm,
    fres: Dataset | None,
    algo: Algorithm | None,
    output: Any,
    data: np.ndarray,
    lonlat: bool,
    min_rotor_diameter: float | None = None,
) -> tuple[float, float] | None:
    """Derive a bounded figure aspect ratio from finite layout extents.

    Parameters
    ----------
    farm
        Farm whose boundary contributes to non-geographic plot bounds.
    fres
        Farm result context; not read or mutated.
    algo
        Algorithm context; not read or mutated.
    output
        Executing output context; not read or mutated.
    data
        Read-only turbine positions with shape ``(n_turbines, 3)``.
    lonlat
        Exclude the projected farm boundary from longitude/latitude extents.
    min_rotor_diameter
        Smallest physical rotor diameter in metres, or ``None`` for no
        minimum pixel-size adjustment.

    Returns
    -------
    figsize
        Width and height in inches, or ``None`` if no finite bounds exist.
        The base area follows Matplotlib defaults and aspect is bounded by two.
    """
    points = data[:, :2]
    points = points[np.all(np.isfinite(points), axis=1)]
    bounds = [points] if len(points) else []
    if not lonlat and farm.boundary is not None:
        boundary = np.stack([farm.boundary.p_min(), farm.boundary.p_max()])
        if np.all(np.isfinite(boundary)):
            bounds.append(boundary)
    if not bounds:
        return None
    extent = np.ptp(np.concatenate(bounds), axis=0)
    max_extent = np.max(extent)
    aspect_extent = np.maximum(extent, max_extent / 2.0)
    aspect = 1.0 if max_extent <= 0.0 else aspect_extent[0] / aspect_extent[1]
    default = np.asarray(plt.rcParams["figure.figsize"], dtype=float)
    base_size = np.sqrt(np.prod(default))
    figsize = base_size * np.array([np.sqrt(aspect), 1.0 / np.sqrt(aspect)])
    if min_rotor_diameter is not None:
        figsize *= _rotor_size_scale(extent, min_rotor_diameter, figsize)
    return float(figsize[0]), float(figsize[1])


def _rotor_size_scale(
    extent: np.ndarray, diameter: float, figsize: np.ndarray
) -> float:
    """Calculate enlargement needed to keep the smallest rotor visible.

    Parameters
    ----------
    extent
        Two plot extents in metres.
    diameter
        Minimum positive rotor diameter in metres.
    figsize
        Proposed figure dimensions in inches; not modified.

    Returns
    -------
    scale
        Uniform enlargement, at least one, targeting four pixels per rotor.
    """
    subplot_fraction = np.array(
        [
            plt.rcParams["figure.subplot.right"] - plt.rcParams["figure.subplot.left"],
            plt.rcParams["figure.subplot.top"] - plt.rcParams["figure.subplot.bottom"],
        ]
    )
    margins = np.array([plt.rcParams["axes.xmargin"], plt.rcParams["axes.ymargin"]])
    plot_extent = (extent + diameter) * (1.0 + 2.0 * margins)
    required = (
        plot_extent
        / diameter
        * _MIN_ROTOR_DIAMETER_PIXELS
        / (float(plt.rcParams["figure.dpi"]) * subplot_fraction)
    )
    return max(1.0, np.max(required / figsize))


def _get_turbine_diameters(
    farm: WindFarm, fres: Dataset | None, algo: Algorithm | None, output: Any
) -> np.ndarray:
    """Resolve finite positive rotor diameters in farm turbine order.

    Parameters
    ----------
    farm
        Farm supplying turbine geometry when results are not selected.
    fres
        Read-only farm results; ``FV.D`` is preferred when
        ``output.from_results`` is enabled.
    algo
        Algorithm resolving turbine-type diameters when results are unavailable.
    output
        Output supplying ``from_results``, positional ``results_state`` and
        fallback ``D`` for unresolved turbine-instance diameters.

    Returns
    -------
    diameters
        Values in metres with shape ``(n_turbines,)``. A single value is
        expanded to every turbine. Arrays may share result storage.

    Raises
    ------
    ValueError
        If the number of diameters differs from turbine count, or any diameter
        is nonfinite or nonpositive.
    """
    if output.from_results and fres is not None and FV.D in fres:
        values = fres[FV.D]
        if FC.STATE in values.dims:
            values = values.isel({FC.STATE: output.results_state})
        diameters = np.asarray(values, dtype=config.dtype_double).reshape(-1)
    elif algo is not None:
        diameters = farm.get_rotor_diameters(algo)
    else:
        diameters = np.asarray(
            [np.nan if turbine.D is None else turbine.D for turbine in farm.turbines],
            dtype=config.dtype_double,
        )
        if output.D is not None:
            diameters[~np.isfinite(diameters)] = output.D
    if diameters.size == 1 and farm.n_turbines != 1:
        diameters = np.full(farm.n_turbines, diameters.item())
    if diameters.shape != (farm.n_turbines,):
        raise ValueError(
            f"Expected {farm.n_turbines} rotor diameters, got shape {diameters.shape}"
        )
    if np.any(~np.isfinite(diameters)) or np.any(diameters <= 0.0):
        raise ValueError("True turbine radii require finite positive rotor diameters")
    return diameters


def _add_turbine_circles(
    farm: WindFarm,
    fres: Dataset | None,
    algo: Algorithm | None,
    output: Any,
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    radii: np.ndarray,
    colors: Any,
    **kwargs: Any,
) -> PatchCollection:
    """Add physical turbine disks without changing caller-owned coordinates.

    Parameters
    ----------
    farm
        Farm context; not read or mutated.
    fres
        Farm result context; not read or mutated.
    algo
        Algorithm context; not read or mutated.
    output
        Executing output context; not read or mutated.
    ax
        Axes receiving the circle collection.
    x
        Horizontal turbine coordinates, shape ``(n_turbines,)``.
    y
        Vertical turbine coordinates, shape ``(n_turbines,)``.
    radii
        Physical radii in the same units as coordinates.
    colors
        Direct Matplotlib colors or numeric values for colormap normalization.
    kwargs
        Patch-collection options; ``vmin`` and ``vmax`` set numeric color limits.

    Returns
    -------
    collection
        Circle collection added to ``ax``; numeric color values may share input
        storage and are not modified.
    """
    vmin, vmax = kwargs.pop("vmin", None), kwargs.pop("vmax", None)
    kwargs.setdefault("edgecolors", "face")
    circles = [Circle((xi, yi), radius) for xi, yi, radius in zip(x, y, radii)]
    numeric = (
        colors is not None
        and not isinstance(colors, str)
        and np.issubdtype(np.asarray(colors).dtype, np.number)
    )
    if colors is not None and not numeric:
        kwargs["facecolors"] = colors
    collection = PatchCollection(circles, **kwargs)
    if numeric:
        collection.set_array(np.asarray(colors))
        collection.set_clim(vmin, vmax)
    ax.add_collection(collection)
    return collection


def _color_values(farm: WindFarm, fres: Dataset | None, color_by: str) -> Any:
    """Resolve direct, categorical or state-reduced turbine color values.

    Parameters
    ----------
    farm
        Farm supplying wind-farm and cluster labels.
    fres
        Read-only farm dataset required for every ``color_by`` mode.
    color_by
        Turbine-only field, ``FC.FARM``, ``FC.CLUSTER``, or a field prefixed
        with ``mean_``, ``sum_``, ``min_`` or ``max_``.

    Returns
    -------
    colors
        One value per turbine. Means use literal state weights; the other
        reductions use ordinary NumPy NaN propagation.

    Raises
    ------
    ValueError
        If results are absent or state-weight dimensions are unsupported.
    KeyError
        If the color selection is unknown.
    """
    if fres is None:
        raise ValueError(f"Missing farm_results for color_by '{color_by}'")
    if color_by in fres and fres[color_by].dims == (FC.TURBINE,):
        return fres[color_by]
    if color_by == FC.FARM:
        return farm.wind_farm_list
    if color_by == FC.CLUSTER:
        return farm.cluster_list
    if color_by.startswith("mean_"):
        weights = fres[FV.WEIGHT]
        if weights.dims == (FC.STATE,):
            indices = "s"
        elif weights.dims == (FC.STATE, FC.TURBINE):
            indices = "st"
        else:
            raise ValueError(
                f"Unsupported dimensions for '{FV.WEIGHT}': Expecting '{(FC.STATE,)}' or '{(FC.STATE, FC.TURBINE)}', got '{weights.dims}'"
            )
        return np.einsum(f"st,{indices}->t", fres[color_by[5:]], weights)
    for prefix, reduce in (("sum_", np.sum), ("min_", np.min), ("max_", np.max)):
        if color_by.startswith(prefix):
            return reduce(fres[color_by[len(prefix) :]], axis=0)
    raise KeyError(
        f"Unknown color_by '{color_by}'. Choose: mean_X, sum_X, min_X, max_X, where X is a farm_results variable"
    )


def _reference_markers(
    farm: WindFarm,
    color_by: str | None,
    kwargs: dict,
    title: str | None,
    legend_labels: dict | None,
) -> tuple:
    """Select existing reference-point marker colors and automatic labels.

    Parameters
    ----------
    farm
        Farm containing ordinary turbines and named reference-point turbines.
    color_by
        Explicit field coloring, disabling automatic reference-point colors.
    kwargs
        Caller marker options, copied before defaults are applied.
    title
        Explicit title, or ``None`` for automatic reference-point suppression.
    legend_labels
        Explicit color-to-label mapping, or ``None`` for automatic labels.

    Returns
    -------
    options
        Copied marker options, with reference points red and other turbines
        orange unless explicit colors or field coloring are supplied.
    resolved_title
        Explicit title or empty text for automatically colored reference points.
    resolved_legend
        Original or newly created color-to-label mapping.
    """
    options: dict[str, Any] = {"c": "orange"}
    options.update(kwargs)
    reference = np.array(
        [
            turbine.name is not None
            and (
                turbine.name == "reference point"
                or turbine.name.startswith("reference point ")
            )
            for turbine in farm.turbines
        ]
    )
    if (
        np.any(reference)
        and color_by is None
        and "c" not in kwargs
        and "color" not in kwargs
    ):
        options["c"] = np.where(reference, "red", "orange")
        if legend_labels is None:
            legend_labels = {}
            if np.any(~reference):
                legend_labels["orange"] = "farm turbines"
            legend_labels["red"] = (
                "reference point" if np.sum(reference) == 1 else "reference points"
            )
        if title is None:
            title = ""
    return options, title, legend_labels


def _plot_markers(
    farm: WindFarm,
    fres: Dataset | None,
    algo: Algorithm | None,
    output: Any,
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    radii: np.ndarray | None,
    colors: Any,
    color_by: str | None,
    options: dict,
) -> tuple[Any, bool]:
    """Draw turbine markers, splitting categorical labels into legend entries.

    Parameters
    ----------
    farm
        Farm context passed to physical-circle drawing.
    fres
        Farm result context passed to physical-circle drawing.
    algo
        Algorithm context passed to physical-circle drawing.
    output
        Executing output context passed to physical-circle drawing.
    ax
        Caller-owned axes receiving markers.
    x
        Horizontal coordinates, shape ``(n_turbines,)``.
    y
        Vertical coordinates, shape ``(n_turbines,)``.
    radii
        Physical radii, or ``None`` to use scatter markers.
    colors
        Direct colors, numeric values or categorical turbine labels.
    color_by
        Coloring field name, or ``None`` for direct colors.
    options
        Marker options, forwarded without modification.

    Returns
    -------
    markers
        Added collection, or the final collection for categorical markers.
    categorical
        Whether a categorical legend was added instead of numeric coloring.
    """
    categorical = not (
        color_by is None
        or colors is None
        or isinstance(colors, str)
        or np.issubdtype(np.asarray(colors).dtype, np.number)
    )
    if not categorical:
        markers = (
            ax.scatter(x, y, c=colors, **options)
            if radii is None
            else _add_turbine_circles(
                farm, fres, algo, output, ax, x, y, radii, colors, **options
            )
        )
        return markers, False
    labels = np.array(colors)
    assert labels.shape == (len(x),), (
        f"Expecting color_by variable with shape {(len(x),)}, got {labels.shape}"
    )
    for label in np.unique(labels):
        selected = labels == label
        if radii is None:
            markers = ax.scatter(
                x[selected], y[selected], c=colors[selected], label=label, **options
            )
        else:
            markers = _add_turbine_circles(
                farm,
                fres,
                algo,
                output,
                ax,
                x[selected],
                y[selected],
                radii[selected],
                colors[selected],
                label=label,
                **options,
            )
        ax.legend(title=color_by, loc="center left", bbox_to_anchor=(1, 0.5))
    return markers, True


def _annotate_turbines(
    farm: WindFarm,
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    annotate: int,
    fontsize: int,
    dx: float,
    dy: float,
) -> None:
    """Add turbine or wind-farm annotations in plotted coordinate units.

    Parameters
    ----------
    farm
        Farm supplying turbine names and wind-farm membership.
    ax
        Axes receiving annotations.
    x
        Horizontal turbine positions, shape ``(n_turbines,)``.
    y
        Vertical turbine positions, shape ``(n_turbines,)``.
    annotate
        ``1`` labels positional indices, ``2`` names, ``3`` wind-farm centres;
        other values omit annotations.
    fontsize
        Annotation size in points.
    dx
        Horizontal offset for turbine labels.
    dy
        Vertical offset for turbine labels.
    """
    if annotate == 1:
        for index in range(len(x)):
            ax.annotate(index, (x[index] + dx, y[index] + dy), size=fontsize)
    elif annotate == 2:
        for index, turbine in enumerate(farm.turbines):
            ax.annotate(turbine.name, (x[index] + dx, y[index] + dy), size=fontsize)
    elif annotate == 3:
        for name, indices in farm.get_wind_farm_mapping().items():
            ax.text(np.mean(x[indices]), np.mean(y[indices]), name, dict(size=fontsize))


def _layout_diameters(
    farm: WindFarm,
    fres: Dataset | None,
    algo: Algorithm | None,
    output: Any,
    normalize_D: bool,
    true_turbine_radii: bool,
) -> tuple[Any, np.ndarray | None]:
    """Resolve layout normalization and physical-marker diameters.

    Parameters
    ----------
    farm
        Farm supplying ordered turbine geometry.
    fres
        Read-only farm results used when results-based geometry is enabled.
    algo
        Algorithm resolving missing turbine-instance diameters.
    output
        Output supplying the explicit normalization diameter and source options.
    normalize_D
        Require uniform diameters if no explicit diameter is configured.
    true_turbine_radii
        Resolve every rotor diameter for physical circle markers.

    Returns
    -------
    normalization
        Explicit or resolved uniform diameter, or ``None`` if not required.
    diameters
        Per-turbine diameters in metres, or ``None`` if not required.

    Raises
    ------
    ValueError
        If required diameters are invalid or normalization needs nonuniform
        diameters.
    """
    diameter = output.D
    diameters = None
    if farm.n_turbines:
        if true_turbine_radii or (normalize_D and diameter is None):
            diameters = _get_turbine_diameters(farm, fres, algo, output)
        if normalize_D and diameter is None:
            assert diameters is not None
            if np.min(diameters) != np.max(diameters):
                raise ValueError(f"Expecting uniform D, found {diameters}")
            diameter = diameters[0]
    return diameter, diameters


def _layout_axes(
    farm: WindFarm,
    fres: Dataset | None,
    algo: Algorithm | None,
    output: Any,
    data: np.ndarray,
    lonlat: bool,
    min_diameter: float | None,
    figsize: Any,
    fig: Any,
    ax: Axes | None,
) -> tuple[Any, Axes]:
    """Create layout axes or reuse the supplied figure's axes without clearing.

    Parameters
    ----------
    farm
        Farm supplying the boundary for automatic figure sizing.
    fres
        Farm result context passed to automatic sizing.
    algo
        Algorithm context passed to automatic sizing.
    output
        Executing output context passed to automatic sizing.
    data
        Read-only turbine positions with shape ``(n_turbines, 3)``.
    lonlat
        Exclude projected boundaries from geographic extents.
    min_diameter
        Minimum physical rotor diameter in metres, or ``None`` for no scaling.
    figsize
        New figure dimensions in inches; ``None`` derives them from extents.
    fig
        Figure to reuse; ``None`` creates a new figure and subplot.
    ax
        Caller-owned axes used only when ``fig`` is supplied; ``None`` then
        selects that figure's first axes.

    Returns
    -------
    figure
        Supplied or newly created figure.
    axes
        Supplied, first existing, or newly created axes.
    """
    if fig is None:
        if figsize is None:
            figsize = _get_auto_figsize(
                farm,
                fres,
                algo,
                output,
                data,
                lonlat,
                min_rotor_diameter=min_diameter,
            )
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        ax = fig.axes[0] if ax is None else ax
    return fig, ax
