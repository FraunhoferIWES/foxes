"""Build static slice coordinates in the farm or wind-aligned reference frame."""

from __future__ import annotations

from typing import Any

import numpy as np
from xarray import Dataset

import foxes.variables as FV
from foxes.config import config
from foxes.utils import wd2uv


def _reference_coordinates(
    farm_results: Dataset, x_direction: float
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Project turbine positions onto a wind-aligned orthonormal frame.

    Parameters
    ----------
    farm_results
        Read-only ``FV.X``, ``FV.Y`` and ``FV.H`` on ``(FC.STATE, FC.TURBINE)``.
    x_direction
        Wind direction defining the x axis in degrees, with zero at north.

    Returns
    -------
    positions
        Newly allocated projected positions with shape
        ``(3, n_states, n_turbines)`` in metres.
    basis
        Unit vectors for the projected x, y and vertical z axes.
    """
    n_states, n_turbines = farm_results[FV.H].shape
    n_x = np.append(wd2uv(x_direction), [0.0], axis=0)
    n_z = np.array([0.0, 0.0, 1.0])
    n_y = np.cross(n_z, n_x)
    xyz = np.zeros((n_states, n_turbines, 3), dtype=config.dtype_double)
    for axis, variable in enumerate((FV.X, FV.Y, FV.H)):
        xyz[:, :, axis] = farm_results[variable]
    return (
        np.array([np.einsum("std,d->st", xyz, axis) for axis in (n_x, n_y, n_z)]),
        (n_x, n_y, n_z),
    )


def _grid_axes(
    lower: tuple | list, upper: tuple | list, resolution: Any, n_img_points: Any
) -> tuple:
    """Construct two inclusive grid axes with the established spacing rule.

    Parameters
    ----------
    lower
        Lower bounds in metres for the two varying axes.
    upper
        Corresponding upper bounds in metres.
    resolution
        Requested spacing in metres, mutually exclusive with ``n_img_points``.
    n_img_points
        Two axis lengths, mutually exclusive with ``resolution``.

    Returns
    -------
    first
        First coordinate vector and its ``linspace`` step.
    second
        Second coordinate vector and its ``linspace`` step.

    Raises
    ------
    ValueError
        If neither or both grid-size specifications are supplied.
    """
    if resolution is not None and n_img_points is None:
        counts = [int((hi - lo) / resolution + 0.5) + 1 for lo, hi in zip(lower, upper)]
    elif resolution is None and n_img_points is not None:
        counts = n_img_points
    else:
        raise ValueError(
            "Expecting either 'resolution' or 'n_img_points', got: "
            f"resolution = {resolution}, n_img_points = {n_img_points}"
        )
    n_a, n_b = counts
    return (
        np.linspace(
            lower[0], upper[0], num=n_a, endpoint=True, retstep=True, dtype=None
        ),
        np.linspace(
            lower[1], upper[1], num=n_b, endpoint=True, retstep=True, dtype=None
        ),
    )


def _report_grid(
    plane: str, lower: list, upper: list, positions: list, steps: tuple, n_points: int
) -> None:
    """Print grid diagnostics using the established verbosity output.

    Parameters
    ----------
    plane
        Names of the two varying axes.
    lower
        Lower x, y and z bounds in metres.
    upper
        Upper x, y and z bounds in metres.
    positions
        Coordinate vectors and the fixed coordinate in x, y, z order.
    steps
        Spacing of the varying coordinates.
    n_points
        Total number of static grid points.
    """
    fixed = next(axis for axis in "xyz" if axis not in plane)
    indices = ["xyz".index(axis) for axis in plane]
    print("\nFlowPlots2D plot grid:")
    print("Min XYZ  =", *lower)
    print("Max XYZ  =", *upper)
    print(f"Pos {fixed.upper()}    =", positions["xyz".index(fixed)])
    print(f"Res {plane.upper()}   =", *steps)
    print(f"Dim {plane.upper()}   =", *(len(positions[i]) for i in indices))
    print("Grid pts =", n_points)


def _vertical_grid(
    farm_results: Dataset,
    plane: str,
    resolution: Any,
    n_img_points: Any,
    x_direction: float,
    lower: list,
    upper: list,
    fixed: Any,
    space: tuple,
    verbosity: int,
) -> tuple:
    """Build an xz or yz grid without duplicating coordinates over states.

    Parameters
    ----------
    farm_results
        Selected, read-only farm coordinates on ``(FC.STATE, FC.TURBINE)``.
    plane
        ``"xz"`` or ``"yz"``.
    resolution
        Requested spacing in metres, exclusive with ``n_img_points``.
    n_img_points
        Counts along the two varying axes.
    x_direction
        Projected x-axis wind direction in degrees.
    lower
        Lower varying-axis bounds; ``None`` derives a bound from the farm.
    upper
        Upper varying-axis bounds; ``None`` derives a bound from the farm.
    fixed
        Fixed-axis coordinate in metres; ``None`` uses its farm-range midpoint.
    space
        Horizontal and vertical margins in metres for automatic bounds.
    verbosity
        Print grid diagnostics above level one.

    Returns
    -------
    grid
        ``(x_pos, y_pos, z_pos, points)``. Two positions are vectors and one
        is scalar. Newly allocated points have shape ``(n_points, 3)`` in
        global metres, with the second varying axis changing fastest.
    """
    coordinates, basis = _reference_coordinates(farm_results, x_direction)
    horizontal = 0 if plane == "xz" else 1
    other = 1 - horizontal
    lower = list(lower)
    upper = list(upper)
    lower[0] = (
        lower[0] if lower[0] is not None else np.min(coordinates[horizontal]) - space[0]
    )
    lower[1] = (
        lower[1]
        if lower[1] is not None
        else np.minimum(np.min(coordinates[2]) - space[1], 0.0)
    )
    upper[0] = (
        upper[0] if upper[0] is not None else np.max(coordinates[horizontal]) + space[0]
    )
    upper[1] = upper[1] if upper[1] is not None else np.max(coordinates[2]) + space[1]
    fixed_min = fixed if fixed is not None else np.min(coordinates[other])
    fixed_max = fixed if fixed is not None else np.max(coordinates[other])
    (a_pos, a_step), (b_pos, b_step) = _grid_axes(
        lower, upper, resolution, n_img_points
    )
    positions: list[Any] = [None, None, b_pos]
    positions[horizontal] = a_pos
    positions[other] = 0.5 * (fixed_min + fixed_max)
    points = np.zeros((len(a_pos), len(b_pos), 3), dtype=config.dtype_double)
    # Keep x, y, z accumulation order to preserve floating-point coordinates.
    for axis, vector in enumerate(basis):
        pos = positions[axis]
        if axis == horizontal:
            points[:] += pos[:, None, None] * vector[None, None, :]
        elif axis == 2:
            points[:] += pos[None, :, None] * vector[None, None, :]
        else:
            points[:] += pos * vector[None, None, :]
    n_points = len(a_pos) * len(b_pos)
    if verbosity > 1:
        low_xyz, high_xyz = [None] * 3, [None] * 3
        low_xyz[horizontal], low_xyz[2], low_xyz[other] = lower[0], lower[1], fixed_min
        high_xyz[horizontal], high_xyz[2], high_xyz[other] = (
            upper[0],
            upper[1],
            fixed_max,
        )
        _report_grid(plane, low_xyz, high_xyz, positions, (a_step, b_step), n_points)
    return (*positions, points.reshape(n_points, 3))
