from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import Delaunay, QhullError, cKDTree


GridAxes = tuple[np.ndarray, ...]


def _validated_points(points: ArrayLike) -> np.ndarray:
    """Validate explicit three-dimensional support points."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError(
            f"Expecting non-empty points with shape (n_points, 3), got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Grid support points must be finite")
    return values


def _validated_axes(axes: Sequence[ArrayLike]) -> GridAxes:
    """Validate strictly increasing grid axes."""
    output = []
    for index, axis in enumerate(axes):
        values = np.asarray(axis, dtype=np.float64)
        if values.ndim != 1 or not len(values):
            raise ValueError(f"Grid axis {index} must be a non-empty 1D array")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Grid axis {index} must contain only finite values")
        if len(values) > 1 and np.any(np.diff(values) <= 0.0):
            raise ValueError(f"Grid axis {index} must be strictly increasing")
        output.append(values)
    if len(output) < 2:
        raise ValueError(f"Expecting at least two grid axes, got {len(output)}")
    return tuple(output)


def _validated_bounds(
    bounds: tuple[ArrayLike, ArrayLike], dimensions: int
) -> tuple[np.ndarray, np.ndarray]:
    """Validate lower and upper grid-selection bounds."""
    lower = np.asarray(bounds[0], dtype=np.float64)
    upper = np.asarray(bounds[1], dtype=np.float64)
    expected = (dimensions,)
    if lower.shape != expected or upper.shape != expected:
        raise ValueError(
            f"Grid bounds must both have shape {expected}, got "
            f"{lower.shape} and {upper.shape}"
        )
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("Grid bounds must be finite")
    if np.any(lower >= upper):
        raise ValueError(
            f"Grid lower bounds must be below upper bounds, got {lower} and {upper}"
        )
    return lower, upper


def detect_regular_grid(points: ArrayLike) -> GridAxes | None:
    """
    Detect whether points form one complete Cartesian three-dimensional grid.

    Duplicate points are ignored. The returned axes are sorted ascending.

    Parameters
    ----------
    points
        Explicit point coordinates with shape ``(n_points, 3)``.

    Returns
    -------
    axes
        The three Cartesian axes, or ``None`` if any axis combination is
        missing.

    Examples
    --------
    >>> points = np.array([[0, 0, 10], [0, 1, 10], [1, 0, 10], [1, 1, 10]])
    >>> tuple(axis.tolist() for axis in detect_regular_grid(points))
    ([0.0, 1.0], [0.0, 1.0], [10.0])
    """
    unique_points = np.unique(_validated_points(points), axis=0)
    axes = tuple(np.unique(unique_points[:, index]) for index in range(3))
    expected = int(np.prod([len(axis) for axis in axes]))
    return axes if len(unique_points) == expected else None


def select_grid_axes(
    axes: Sequence[ArrayLike],
    bounds: tuple[ArrayLike, ArrayLike] | None = None,
    padding: int = 1,
) -> GridAxes:
    """
    Select grid axes within bounds plus surrounding native grid points.

    Bounds apply to the first dimensions of ``axes``. For example, two-value
    bounds crop horizontal axes while preserving a third height axis.

    Parameters
    ----------
    axes
        Strictly increasing Cartesian coordinate axes.
    bounds
        Optional lower and upper coordinate bounds.
    padding
        Number of native grid points required outside each bound.

    Returns
    -------
    selected_axes
        The selected Cartesian axes.
    """
    selected = list(_validated_axes(axes))
    if not isinstance(padding, int) or padding < 0:
        raise ValueError(f"Grid padding must be a non-negative integer, got {padding}")
    if bounds is None:
        return tuple(selected)
    lower_values = np.asarray(bounds[0])
    if lower_values.ndim != 1 or not len(lower_values):
        raise ValueError(
            f"Grid lower bounds must be a non-empty 1D array, got {lower_values.shape}"
        )
    dimensions = len(lower_values)
    if dimensions > len(selected):
        raise ValueError(
            f"Grid bounds have {dimensions} dimensions, but there are only "
            f"{len(selected)} axes"
        )
    lower, upper = _validated_bounds(bounds, dimensions)
    for index in range(dimensions):
        axis = selected[index]
        start = int(np.searchsorted(axis, lower[index], side="left")) - padding
        stop = int(np.searchsorted(axis, upper[index], side="right")) + padding
        if start < 0 or stop > len(axis):
            raise ValueError(
                f"Grid axis {index} has fewer than {padding} exterior points "
                f"around bounds [{lower[index]}, {upper[index]}]"
            )
        selected[index] = axis[start:stop]
    return tuple(selected)


def _infer_resolution(points: np.ndarray) -> float:
    """Infer horizontal spacing from median nearest-neighbour distance."""
    horizontal = np.unique(points[:, :2], axis=0)
    if len(horizontal) < 3:
        raise ValueError("At least three horizontal support points are required")
    distances = cKDTree(horizontal).query(horizontal, k=2)[0][:, 1]
    resolution = float(np.median(distances[distances > 0.0]))
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("Could not infer a positive grid resolution")
    return resolution


def _regular_axis(
    lower: float, upper: float, resolution: float, padding: int
) -> np.ndarray:
    """Create a regular axis around limits with interval padding."""
    start = np.floor(lower / resolution) * resolution - padding * resolution
    stop = np.ceil(upper / resolution) * resolution + padding * resolution
    count = int(np.round((stop - start) / resolution)) + 1
    return start + resolution * np.arange(count)


def _inside_hull(axes: GridAxes, support: np.ndarray) -> bool:
    """Check whether rectangular horizontal grid corners lie in a convex hull."""
    try:
        hull = Delaunay(support)
    except QhullError as exc:
        raise ValueError("Grid support does not span a two-dimensional area") from exc
    x, y = axes[:2]
    corners = np.array([[x[0], y[0]], [x[0], y[-1]], [x[-1], y[0]], [x[-1], y[-1]]])
    return bool(np.all(hull.find_simplex(corners) >= 0))


def regular_grid_from_points(
    points: ArrayLike,
    bounds: tuple[ArrayLike, ArrayLike] | None = None,
    padding: int = 1,
) -> tuple[GridAxes, bool]:
    """
    Detect a Cartesian grid or select one for irregular support points.

    Complete Cartesian support retains its native axes. For irregular support,
    horizontal spacing is the median nearest-neighbour distance. Without
    bounds, the largest centred rectangle found inside the support convex hull
    is returned. With bounds, the selected grid surrounds them by ``padding``
    intervals and must remain inside the support convex hull. Heights are the
    sorted unique support heights.

    Parameters
    ----------
    points
        Explicit point coordinates with shape ``(n_points, 3)``.
    bounds
        Optional lower and upper horizontal bounds, each with shape ``(2,)``.
    padding
        Number of native points or inferred grid intervals outside each bound.

    Returns
    -------
    axes
        The selected ``(x, y, height)`` grid axes.
    source_is_grid
        Whether the source points form a complete Cartesian grid.
    """
    values = _validated_points(points)
    if not isinstance(padding, int) or padding < 0:
        raise ValueError(f"Grid padding must be a non-negative integer, got {padding}")
    axes = detect_regular_grid(values)
    if axes is not None:
        return select_grid_axes(axes, bounds=bounds, padding=padding), True

    horizontal = np.unique(values[:, :2], axis=0)
    resolution = _infer_resolution(values)
    heights = np.unique(values[:, 2])
    if bounds is not None:
        lower, upper = _validated_bounds(bounds, 2)
        x = _regular_axis(lower[0], upper[0], resolution, padding)
        y = _regular_axis(lower[1], upper[1], resolution, padding)
    else:
        lower = np.min(horizontal, axis=0)
        upper = np.max(horizontal, axis=0)
        x = np.arange(lower[0], upper[0] + 0.5 * resolution, resolution)
        y = np.arange(lower[1], upper[1] + 0.5 * resolution, resolution)
        while len(x) and len(y) and not _inside_hull((x, y, heights), horizontal):
            x = x[1:-1]
            y = y[1:-1]

    axes = (x, y, heights)
    if not len(x) or not len(y) or not _inside_hull(axes, horizontal):
        raise ValueError("Could not place a regular grid inside point support")
    return axes, False
