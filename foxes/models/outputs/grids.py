"""Static output grids and conversions between gridded result representations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV
from foxes.config import config
from foxes.utils import wd2uv

from ._grid_data import data2xr as data2xr
from ._grid_geometry import _grid_axes, _report_grid, _vertical_grid

if TYPE_CHECKING:
    from foxes.core import Algorithm


def project_points(
    points: np.ndarray, plane: str, x_direction: float = 270
) -> np.ndarray:
    """Project coordinates onto the configured slice reference axes.

    Parameters
    ----------
    points
        Global coordinates in metres, with trailing dimension three.
    plane
        ``"xy"`` leaves coordinates unchanged; ``"xz"`` and ``"yz"`` use
        the wind-aligned frame.
    x_direction
        Wind direction of the projected x axis in degrees, zero at north.

    Returns
    -------
    projected
        Coordinates with the input shape. The xy result is the supplied array;
        vertical-plane results are newly allocated. Inputs are not modified.

    Raises
    ------
    ValueError
        If ``plane`` is not ``"xy"``, ``"xz"`` or ``"yz"``.
    """
    if plane == "xy":
        return points
    if plane not in ("xz", "yz"):
        raise ValueError(f"Unknown slice plane '{plane}'")
    x_axis = np.append(wd2uv(x_direction), 0)
    y_axis = np.cross([0, 0, 1], x_axis)
    return np.stack([points @ x_axis, points @ y_axis, points[..., 2]], axis=-1)


def _select_states(
    farm_results: Dataset,
    states_sel: Any,
    states_isel: Any,
) -> Dataset:
    """Select states by label, resolving a positional request when supplied.

    Parameters
    ----------
    farm_results
        Read-only dataset with an ``FC.STATE`` coordinate.
    states_sel
        Labels accepted by xarray ``sel``, or ``None`` for no label selection.
    states_isel
        Positions accepted by the state index, exclusive with ``states_sel``.

    Returns
    -------
    selected
        Selected dataset, potentially sharing input storage; the original
        object is returned if no selection is requested.

    Raises
    ------
    ValueError
        If both selections are supplied, or positional selection cannot be
        converted to labels because state labels are duplicated.
    """
    if states_sel is not None and states_isel is not None:
        raise ValueError("Choose either 'states_sel' or 'states_isel', not both")
    if states_isel is not None:
        state_index = farm_results.get_index(FC.STATE)
        if not state_index.is_unique:
            raise ValueError(
                "Cannot convert 'states_isel' to labels because the state "
                "coordinate contains duplicates"
            )
        states_sel = state_index[states_isel].tolist()
    return (
        farm_results.sel({FC.STATE: states_sel})
        if states_sel is not None
        else farm_results
    )


def calc_point_results(
    algo: Algorithm,
    g_pts: np.ndarray,
    farm_results: Dataset | None = None,
    seq_iter: Any = None,
    verbosity: int = 0,
    **kwargs: Any,
) -> Dataset:
    """Calculate point results at static grid points.

    Parameters
    ----------
    algo
        Algorithm used for point calculation. Its verbosity is temporarily
        changed and restored after a successful calculation.
    g_pts
        The static grid points, shape: ``(n_pts, 3)``.
    farm_results
        Read-only farm results passed to ``algo.calc_points``.
    seq_iter
        Iterator supplying farm results instead of ``farm_results`` when set.
    verbosity
        The verbosity level, 0 = silent
    kwargs
        Additional parameters for algo.calc_points

    Returns
    -------
    point_results
        Calculated variables with dimensions ``(FC.STATE, FC.POINT)``.

    """
    averb = None if verbosity == algo.verbosity else algo.verbosity
    if averb is not None:
        algo.verbosity = verbosity
    fres = farm_results if seq_iter is None else seq_iter.farm_results
    point_results = algo.calc_points(fres, points=g_pts, **kwargs)
    if averb is not None:
        algo.verbosity = averb

    return point_results


def get_grid_xy(
    farm_results: Dataset,
    resolution: Any = None,
    n_img_points: Any = None,
    xmin: Any = None,
    ymin: Any = None,
    xmax: Any = None,
    ymax: Any = None,
    z: Any = None,
    xspace: float = 500.0,
    yspace: float = 500.0,
    states_sel: Any = None,
    states_isel: Any = None,
    verbosity: int = 0,
) -> tuple[np.ndarray, np.ndarray, Any, np.ndarray]:
    """Generate a static coordinate grid in a horizontal xy-plane.

    Parameters
    ----------
    farm_results
        The farm results. The calculated variables have
        dimensions ``(FC.STATE, FC.TURBINE)``; inputs are not modified.
    resolution
        Requested spacing in metres, mutually exclusive with ``n_img_points``.
    n_img_points
        Number of coordinates ``(n_x, n_y)``, exclusive with ``resolution``.
    xmin
        The min x coordinate, or None for automatic
    ymin
        The min y coordinate, or None for automatic
    xmax
        The max x coordinate, or None for automatic
    ymax
        The max y coordinate, or None for automatic
    z
        Fixed height in metres; ``None`` uses the midpoint of hub-height bounds.
    xspace
        The extra space in x direction, before and after wind farm
    yspace
        The extra space in y direction, before and after wind farm
    states_sel
        Reduce to selected states
    states_isel
        Reduce to the selected states indices
    verbosity
        The verbosity level

    Returns
    -------
    x_pos
        New x coordinates in metres, shape ``(n_x,)``.
    y_pos
        New y coordinates in metres, shape ``(n_y,)``.
    z_pos
        The z position of the grid
    g_pts
        The static grid points, shape: ``(n_pts, 3)``. Coordinates are
        shared by all selected states.

    Raises
    ------
    ValueError
        If both or neither grid-size options are given, both state selections
        are given, or positional state selection encounters duplicate labels.

    """
    farm_results = _select_states(farm_results, states_sel, states_isel)

    x_min = xmin if xmin is not None else farm_results[FV.X].min().to_numpy() - xspace
    y_min = ymin if ymin is not None else farm_results[FV.Y].min().to_numpy() - yspace
    z_min = z if z is not None else farm_results[FV.H].min().to_numpy()
    x_max = xmax if xmax is not None else farm_results[FV.X].max().to_numpy() + xspace
    y_max = ymax if ymax is not None else farm_results[FV.Y].max().to_numpy() + yspace
    z_max = z if z is not None else farm_results[FV.H].max().to_numpy()

    (x_pos, x_res), (y_pos, y_res) = _grid_axes(
        (x_min, y_min), (x_max, y_max), resolution, n_img_points
    )
    n_pts = len(x_pos) * len(y_pos)
    z_pos = 0.5 * (z_min + z_max)
    g_pts: np.ndarray = np.zeros((len(x_pos), len(y_pos), 3), dtype=config.dtype_double)
    g_pts[:, :, 0] = x_pos[:, None]
    g_pts[:, :, 1] = y_pos[None, :]
    g_pts[:, :, 2] = z_pos

    if verbosity > 1:
        _report_grid(
            "xy",
            [x_min, y_min, z_min],
            [x_max, y_max, z_max],
            [x_pos, y_pos, z_pos],
            (x_res, y_res),
            n_pts,
        )

    return (
        x_pos,
        y_pos,
        z_pos,
        g_pts.reshape(n_pts, 3),
    )


def get_grid_xz(
    farm_results: Dataset,
    resolution: Any = None,
    n_img_points: Any = None,
    x_direction: Any = 270,
    xmin: Any = None,
    zmin: float = 0.0,
    xmax: Any = None,
    zmax: Any = None,
    y: Any = None,
    xspace: float = 500.0,
    zspace: float = 500.0,
    states_sel: Any = None,
    states_isel: Any = None,
    verbosity: int = 0,
) -> tuple[np.ndarray, Any, np.ndarray, np.ndarray]:
    """Generate a static coordinate grid in a vertical xz-plane.

    Parameters
    ----------
    farm_results
        The farm results. The calculated variables have
        dimensions ``(FC.STATE, FC.TURBINE)``; inputs are not modified.
    resolution
        Requested spacing in metres, mutually exclusive with ``n_img_points``.
    n_img_points
        Number of coordinates ``(n_x, n_z)``, exclusive with ``resolution``.
    x_direction
        Wind direction of the projected x axis in degrees, zero at north.
    xmin
        The min x coordinate, or None for automatic
    zmin
        Minimum height in metres; ``None`` derives it from hub heights and
        ``zspace``, capped at zero.
    xmax
        The max x coordinate, or None for automatic
    zmax
        The max z coordinate, or None for automatic
    y
        Fixed projected y coordinate in metres; ``None`` uses its farm-range
        midpoint.
    xspace
        The extra space in x direction, before and after wind farm
    zspace
        The extra space in z direction, below and above wind farm
    states_sel
        Reduce to selected states
    states_isel
        Reduce to the selected states indices
    verbosity
        The verbosity level

    Returns
    -------
    x_pos
        New projected x coordinates in metres, shape ``(n_x,)``.
    y_pos
        The y position of the grid
    z_pos
        New height coordinates in metres, shape ``(n_z,)``.
    g_pts
        The static grid points, shape: ``(n_pts, 3)``. Coordinates are
        shared by all selected states.

    Raises
    ------
    ValueError
        If both or neither grid-size options are given, both state selections
        are given, or positional state selection encounters duplicate labels.

    """

    farm_results = _select_states(farm_results, states_sel, states_isel)
    return _vertical_grid(
        farm_results,
        "xz",
        resolution,
        n_img_points,
        x_direction,
        [xmin, zmin],
        [xmax, zmax],
        y,
        (xspace, zspace),
        verbosity,
    )


def get_grid_yz(
    farm_results: Dataset,
    resolution: Any = None,
    n_img_points: Any = None,
    x_direction: Any = 270,
    ymin: Any = None,
    zmin: float = 0.0,
    ymax: Any = None,
    zmax: Any = None,
    x: Any = None,
    yspace: float = 500.0,
    zspace: float = 500.0,
    states_sel: Any = None,
    states_isel: Any = None,
    verbosity: int = 0,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a static coordinate grid in a vertical yz-plane.

    Parameters
    ----------
    farm_results
        The farm results. The calculated variables have
        dimensions ``(FC.STATE, FC.TURBINE)``; inputs are not modified.
    resolution
        Requested spacing in metres, mutually exclusive with ``n_img_points``.
    n_img_points
        Number of coordinates ``(n_y, n_z)``, exclusive with ``resolution``.
    x_direction
        Wind direction of the projected x axis in degrees, zero at north.
    ymin
        The min y coordinate, or None for automatic
    zmin
        Minimum height in metres; ``None`` derives it from hub heights and
        ``zspace``, capped at zero.
    ymax
        The max y coordinate, or None for automatic
    zmax
        The max z coordinate, or None for automatic
    x
        Fixed projected x coordinate in metres; ``None`` uses its farm-range
        midpoint.
    yspace
        The extra space in y direction, before and after wind farm
    zspace
        The extra space in z direction, below and above wind farm
    states_sel
        Reduce to selected states
    states_isel
        Reduce to the selected states indices
    verbosity
        The verbosity level

    Returns
    -------
    x_pos
        The x position of the grid
    y_pos
        New projected y coordinates in metres, shape ``(n_y,)``.
    z_pos
        New height coordinates in metres, shape ``(n_z,)``.
    g_pts
        The static grid points, shape: ``(n_pts, 3)``. Coordinates are
        shared by all selected states.

    Raises
    ------
    ValueError
        If both or neither grid-size options are given, both state selections
        are given, or positional state selection encounters duplicate labels.

    """

    farm_results = _select_states(farm_results, states_sel, states_isel)
    return _vertical_grid(
        farm_results,
        "yz",
        resolution,
        n_img_points,
        x_direction,
        [ymin, zmin],
        [ymax, zmax],
        x,
        (yspace, zspace),
        verbosity,
    )


def np2np_p(
    data: dict[str, np.ndarray], a_pos: np.ndarray, b_pos: np.ndarray
) -> np.ndarray:
    """Stack point fields into a new variable-last NumPy grid.

    Parameters
    ----------
    data
        Read-only variable-to-values mapping with ``n_a * n_b`` values per
        field, ordered with the second coordinate changing fastest.
    a_pos
        First-axis coordinates, shape ``(n_a,)``; only their length is used.
    b_pos
        Second-axis coordinates, shape ``(n_b,)``; only their length is used.

    Returns
    -------
    out
        New array with shape ``(n_a, n_b, n_variables)`` and configured double
        dtype. The variable axis follows mapping insertion order.

    """
    n_a = len(a_pos)
    n_b = len(b_pos)
    n_v = len(data)
    out: np.ndarray = np.zeros((n_a, n_b, n_v), dtype=config.dtype_double)
    for vi, d in enumerate(data.values()):
        out[:, :, vi] = d.reshape(n_a, n_b)
    return out


def np2np_sp(
    data: dict[str, np.ndarray],
    states: np.ndarray,
    a_pos: np.ndarray,
    b_pos: np.ndarray,
) -> np.ndarray:
    """Stack state-point fields into a new variable-last NumPy grid.

    Parameters
    ----------
    data
        Read-only variable-to-values mapping with ``n_states * n_a * n_b``
        values per field, in state-major C ordering.
    states
        State labels, shape ``(n_states,)``; only their length is used.
    a_pos
        First-axis coordinates, shape ``(n_a,)``.
    b_pos
        Second-axis coordinates, shape ``(n_b,)``.

    Returns
    -------
    out
        New array with shape ``(n_states, n_a, n_b, n_variables)`` and
        configured double dtype. Variables follow mapping insertion order.

    """
    n_s = len(states)
    n_a = len(a_pos)
    n_b = len(b_pos)
    n_v = len(data)
    out: np.ndarray = np.zeros((n_s, n_a, n_b, n_v), dtype=config.dtype_double)
    for vi, d in enumerate(data.values()):
        out[:, :, :, vi] = d.reshape(n_s, n_a, n_b)
    return out


def np2pd_p(
    data: dict[str, np.ndarray],
    a_pos: np.ndarray,
    b_pos: np.ndarray,
    ori: str,
    label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Place flattened grid fields in a position-indexed DataFrame.

    Parameters
    ----------
    data
        Variable-to-flattened-values mapping with ``n_a * n_b`` entries per
        field, in C ordering; inputs are not modified.
    a_pos
        First-axis coordinates, shape ``(n_a,)``; only their length is used.
    b_pos
        Second-axis coordinates, shape ``(n_b,)``; only their length is used.
    ori
        The orientation, 'xy' or 'xz' or 'yz'
    label_map
        Mapping for coordinate index names only; column names are retained.

    Returns
    -------
    out
        New DataFrame with positional axis MultiIndex, not coordinate values.
        Columns retain variable names and mapping order.

    """
    if label_map is None:
        label_map = {}
    a, b = [label_map.get(o, o) for o in ori]
    n_a = len(a_pos)
    n_b = len(b_pos)
    minds = pd.MultiIndex.from_product([range(n_a), range(n_b)], names=[a, b])
    return pd.DataFrame(index=minds, data=data)


def np2pd_sp(
    data: dict[str, np.ndarray],
    states: np.ndarray,
    a_pos: np.ndarray,
    b_pos: np.ndarray,
    ori: str,
    label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Place flattened state-grid fields in a state-and-position DataFrame.

    Parameters
    ----------
    data
        Variable-to-flattened-values mapping with ``n_states * n_a * n_b``
        entries per field in C ordering; inputs are not modified.
    states
        State labels, shape ``(n_states,)``, retained as the outer index.
    a_pos
        First-axis coordinates, shape ``(n_a,)``; only their length is used.
    b_pos
        Second-axis coordinates, shape ``(n_b,)``; only their length is used.
    ori
        The orientation, 'xy' or 'xz' or 'yz'
    label_map
        Mapping for state and coordinate index names; column names are retained.

    Returns
    -------
    out
        New DataFrame with state-label and positional-axis MultiIndex.
        Columns retain variable names and mapping order.

    """
    if label_map is None:
        label_map = {}
    a, b = [label_map.get(o, o) for o in ori]
    s = label_map.get(FC.STATE, FC.STATE)
    n_a = len(a_pos)
    n_b = len(b_pos)
    minds = pd.MultiIndex.from_product(
        [states, range(n_a), range(n_b)], names=[s, a, b]
    )
    return pd.DataFrame(index=minds, data=data)


def np2xr_p(
    data: dict[str, np.ndarray],
    a_pos: np.ndarray,
    b_pos: np.ndarray,
    c_pos: Any,
    ori: str,
    label_map: dict[str, str] | None = None,
) -> Dataset:
    """Reshape point fields into a coordinate-labelled slice dataset.

    Parameters
    ----------
    data
        Read-only variable-to-values mapping, with ``n_a * n_b`` entries per
        field in C ordering.
    a_pos
        First-axis coordinates in metres, shape ``(n_a,)``.
    b_pos
        Second-axis coordinates in metres, shape ``(n_b,)``.
    c_pos
        Fixed coordinate of the omitted axis in metres.
    ori
        The orientation, 'xy' or 'xz' or 'yz'
    label_map
        Coordinate-name mapping. Field names are retained in this conversion.

    Returns
    -------
    out
        New dataset with variable dimensions ``(b, a)`` and a fixed-coordinate
        attribute. Arrays may share storage with inputs; no inputs are mutated.

    """
    if label_map is None:
        label_map = {}
    a, b = [label_map.get(o, o) for o in ori]
    c = list(set("xyz") - set(ori))[0]
    c = label_map.get(c, c)
    n_a = len(a_pos)
    n_b = len(b_pos)
    return Dataset(
        coords={b: b_pos, a: a_pos},
        data_vars={
            v: ((b, a), np.swapaxes(d.reshape(n_a, n_b), 0, 1)) for v, d in data.items()
        },
        attrs={c: float(c_pos)},
    )


def np2xr_sp(
    data: dict[str, np.ndarray],
    states: np.ndarray,
    a_pos: np.ndarray,
    b_pos: np.ndarray,
    c_pos: Any,
    ori: str,
    label_map: dict[str, str] | None = None,
) -> Dataset:
    """Reshape state-point fields into a coordinate-labelled slice dataset.

    Parameters
    ----------
    data
        Read-only variable-to-values mapping, with ``n_states * n_a * n_b``
        entries per field in C ordering.
    states
        State labels, shape ``(n_states,)``, retained in order.
    a_pos
        First-axis coordinates in metres, shape ``(n_a,)``.
    b_pos
        Second-axis coordinates in metres, shape ``(n_b,)``.
    c_pos
        Fixed coordinate of the omitted axis in metres.
    ori
        The orientation, 'xy' or 'xz' or 'yz'
    label_map
        The mapping from original to new field names

    Returns
    -------
    out
        New dataset with variable dimensions ``(FC.STATE, b, a)`` before
        renaming, and a fixed-coordinate attribute. Arrays may share storage
        with inputs; no inputs are mutated.

    """
    if label_map is None:
        label_map = {}
    a, b = [label_map.get(o, o) for o in ori]
    c = list(set("xyz") - set(ori))[0]
    c = label_map.get(c, c)
    s = label_map.get(FC.STATE, FC.STATE)
    n_s = len(states)
    n_a = len(a_pos)
    n_b = len(b_pos)
    return Dataset(
        coords={s: states, b: b_pos, a: a_pos},
        data_vars={
            label_map.get(v, v): (
                (s, b, a),
                np.swapaxes(d.reshape(n_s, n_a, n_b), 1, 2),
            )
            for v, d in data.items()
        },
        attrs={c: float(c_pos)},
    )
