"""Render state-resolved flow slices without retaining runtime model state."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

import foxes.variables as FV

from ._get_fig import get_fig


def _rotor_data(fres: Any, si: int, color: Any) -> dict[str, Any] | None:
    """Collect rotor geometry for one state without changing farm results.

    Parameters
    ----------
    fres
        Farm fields with leading dimensions ``(FC.STATE, FC.TURBINE)``.
    si
        Positional state index.
    color
        Matplotlib rotor color; ``None`` disables the overlay.

    Returns
    -------
    rotor_data
        Geometry arrays indexed by turbine, or ``None``. Arrays may share
        storage with ``fres`` and are only read by the renderer.
    """
    if color is None:
        return None
    try:
        angle = fres[FV.YAW][si]
    except KeyError:
        try:
            angle = fres[FV.AMB_WD][si] + fres[FV.YAWM][si]
        except KeyError:
            angle = fres[FV.AMB_WD][si]
    return {
        "color": color,
        "D": fres[FV.D][si],
        "H": fres[FV.H][si],
        "X": fres[FV.X][si],
        "Y": fres[FV.Y][si],
        "AMB_WD": fres[FV.AMB_WD][si],
        "turb_angle": angle,
    }


def _quiver_data(
    parameters: dict[str, Any],
    data: np.ndarray,
    quiver_n: int | None,
    quiver_pars: dict[str, Any] | None,
) -> Any:
    """Prepare wind vectors in the slice's existing array ordering.

    Parameters
    ----------
    parameters
        Slice metadata containing the ordered ``variables``.
    data
        Values with shape ``(n_states, n_a, n_b, n_variables)``.
    quiver_n
        Grid stride, or ``None`` to omit vectors.
    quiver_pars
        Overrides for Matplotlib quiver options; not mutated.

    Returns
    -------
    quiver
        Stride, copied options, wind directions and wind speeds, or ``None``.
        Wind arrays are views of ``data``.
    """
    options = dict(angles="xy", scale_units="xy", scale=0.05)
    options.update({} if quiver_pars is None else quiver_pars)
    if quiver_n is None:
        return None
    variables = parameters["variables"]
    return (
        quiver_n,
        options,
        data[..., variables.index(FV.WD)],
        data[..., variables.index(FV.WS)],
    )


def _slice_title(
    plane: str, title: Any, si: int, state: Any, parameters: dict, grid: tuple
) -> Any:
    """Resolve a callback or the scientific title for one slice.

    Parameters
    ----------
    plane
        Slice orientation, ``"xy"``, ``"xz"`` or ``"yz"``.
    title
        Explicit title or callback called as ``title(si, state)``.
    si
        Positional state index.
    state
        State coordinate label.
    parameters
        Slice metadata, including ``x_direction`` for vertical planes.
    grid
        ``(x_pos, y_pos, z_pos, points)`` in metres.

    Returns
    -------
    resolved_title
        Explicit, callback-generated, or automatic state-and-plane title.
    """
    if callable(title):
        return title(si, state)
    if title is not None:
        return title
    if plane == "xy":
        return f"State {state}, z =  {int(np.round(grid[2]))} m"
    fixed_axis, fixed_pos = ("y", grid[1]) if plane == "xz" else ("x", grid[0])
    return (
        f"State {state}, x direction = {parameters['x_direction']}°"
        f", {fixed_axis} =  {int(np.round(fixed_pos))} m"
    )


def _gen_states_fig(
    fres: Any,
    output: Any,
    states_data: tuple,
    plane: str,
    title: Any,
    add_bar: bool,
    quiver_n: int | None,
    quiver_pars: dict[str, Any] | None,
    animated: bool,
    rotor_color: Any,
    kwargs: dict[str, Any],
) -> Iterator[Any]:
    """Render a sequence using the established plane-specific animation rules.

    Parameters
    ----------
    fres
        Farm fields with leading dimensions ``(FC.STATE, FC.TURBINE)``.
    output
        Executing output supplying ``nofig``; not mutated.
    states_data
        Metadata, NumPy values, state labels and static grid coordinates.
    plane
        Slice orientation, ``"xy"``, ``"xz"`` or ``"yz"``.
    title
        Title or callback ``title(si, state)``; ``None`` selects automatic text.
    add_bar
        Add a color bar, subject to the existing animation rules.
    quiver_n
        Vector stride, or ``None`` to omit vectors.
    quiver_pars
        Matplotlib quiver overrides; not mutated.
    animated
        Mark artists for animation and reuse color limits where applicable.
    rotor_color
        Rotor overlay color, or ``None`` to omit overlays.
    kwargs
        Keyword arguments forwarded to ``get_fig`` without mutation.

    Yields
    ------
    frame
        The result of ``get_fig`` for each state. When ``output.nofig`` is
        enabled, first yield ``None`` and then continue the existing sequence.

    Raises
    ------
    NotImplementedError
        If the slice data format is not ``"numpy"``.
    """
    if output.nofig:
        yield None
    parameters, data, states, grid = states_data
    if parameters["data_format"] != "numpy":
        raise NotImplementedError(
            f"Only numpy data_format is supported here, got {parameters['data_format']}"
        )
    var = parameters["var"]
    values = data[..., parameters["variables"].index(var)]
    vmin, vmax = parameters["vmin"], parameters["vmax"]
    quiver = _quiver_data(parameters, data, quiver_n, quiver_pars)
    a_pos, b_pos = (
        (grid[0], grid[1])
        if plane == "xy"
        else (grid[0] if plane == "xz" else grid[1], grid[2])
    )
    for si, state in enumerate(states):
        vmin, vmax, add_bar = _animation_limits(
            plane, animated, si, values, vmin, vmax, add_bar
        )
        frame_title = (
            _slice_title(plane, title, si, state, parameters, grid)
            if not animated or callable(title)
            else title
        )
        yield get_fig(
            var=var,
            data=values,
            si=si,
            s=state,
            x_pos=a_pos,
            y_pos=b_pos,
            title=frame_title,
            add_bar=add_bar,
            vmin=vmin,
            vmax=vmax,
            quiv=quiver,
            animated=animated,
            show_rotor_dict=_rotor_data(fres, si, rotor_color),
            rotor_plane=plane,
            **_plane_options(plane, grid),
            **kwargs,
        )


def _animation_limits(
    plane: str,
    animated: bool,
    si: int,
    values: np.ndarray,
    vmin: Any,
    vmax: Any,
    add_bar: bool,
) -> tuple[Any, Any, bool]:
    """Apply the existing horizontal and vertical animation color policies.

    Parameters
    ----------
    plane
        Slice orientation, ``"xy"``, ``"xz"`` or ``"yz"``.
    animated
        Apply frame-to-frame animation policies when enabled.
    si
        Positional frame index.
    values
        Read-only values for all frames and grid points.
    vmin
        Current lower color limit; ``None`` permits automatic selection.
    vmax
        Current upper color limit; ``None`` permits automatic selection.
    add_bar
        Current color-bar selection.

    Returns
    -------
    lower
        Lower limit, derived globally only on the first animated xy frame.
    upper
        Upper limit, derived globally only on the first animated xy frame.
    color_bar
        Disabled after the first animated frame in xy, or in vertical planes
        only when both limits are explicit. NaNs are not ignored in extrema.
    """
    if animated and plane == "xy" and si == 0:
        vmin = vmin if vmin is not None else np.min(values)
        vmax = vmax if vmax is not None else np.max(values)
    if (
        animated
        and si > 0
        and (plane == "xy" or (vmin is not None and vmax is not None))
    ):
        add_bar = False
    return vmin, vmax, add_bar


def _plane_options(plane: str, grid: tuple) -> dict[str, Any]:
    """Select exact rotor-slice filtering and yz horizontal-axis inversion.

    Parameters
    ----------
    plane
        Slice orientation, ``"xy"``, ``"xz"`` or ``"yz"``.
    grid
        x, y, z coordinate vectors/scalar and static grid points in metres.

    Returns
    -------
    options
        New keyword mapping for ``get_fig``. Horizontal slices add no options;
        vertical slices filter turbine centres with zero absolute tolerance.
    """
    options: dict[str, Any] = {}
    if plane != "xy":
        options["rotor_slice"] = {
            "axis": "y" if plane == "xz" else "x",
            "value": grid[1 if plane == "xz" else 0],
            "tol": 0.0,
        }
    if plane == "yz":
        options["invert_axis"] = "x"
    return options


def _gen_states_fig_xy(
    fres: Any,
    output: Any,
    states_data_xy: tuple[Any, Any, Any, Any],
    title: Any = None,
    add_bar: bool = True,
    quiver_n: int | None = None,
    quiver_pars: dict[str, Any] | None = None,
    animated: bool = False,
    rotor_color: Any = None,
    **kwargs: Any,
) -> Iterator[Any]:
    """Generate horizontal flow frames in state-coordinate order.

    Parameters
    ----------
    fres
        Read-only farm fields with dimensions ``(FC.STATE, FC.TURBINE)``.
    output
        Executing output supplying ``nofig``.
    states_data_xy
        ``(parameters, data, states, grid_data)``. NumPy ``data`` has shape
        ``(n_states, n_x, n_y, n_variables)``; grid points are static metres.
    title
        Title or callback ``title(si, state)``; ``None`` labels the state and z.
    add_bar
        Add a color bar; animated frames after the first omit it.
    quiver_n
        Draw a wind vector every nth grid point; ``None`` omits vectors.
    quiver_pars
        Overrides for Matplotlib quiver options; not mutated.
    animated
        Mark artists for animation and fix unspecified limits across all states.
    rotor_color
        Rotor line color, or ``None`` to omit rotors.
    kwargs
        Options for ``get_fig``, including caller-owned ``fig`` and ``ax``.

    Yields
    ------
    frame
        Figure or tuple selected by ``get_fig`` return flags. ``nofig`` first
        yields ``None`` without terminating the sequence.

    Raises
    ------
    NotImplementedError
        If the data format is not ``"numpy"``.
    """
    yield from _gen_states_fig(
        fres,
        output,
        states_data_xy,
        "xy",
        title,
        add_bar,
        quiver_n,
        quiver_pars,
        animated,
        rotor_color,
        kwargs,
    )


def _gen_states_fig_xz(
    fres: Any,
    output: Any,
    states_data_xz: tuple[Any, Any, Any, Any],
    title: Any = None,
    add_bar: bool = True,
    quiver_n: int | None = None,
    quiver_pars: dict[str, Any] | None = None,
    animated: bool = False,
    rotor_color: Any = None,
    **kwargs: Any,
) -> Iterator[Any]:
    """Generate vertical xz flow frames in state-coordinate order.

    Parameters
    ----------
    fres
        Read-only farm fields with dimensions ``(FC.STATE, FC.TURBINE)``.
    output
        Executing output supplying ``nofig``.
    states_data_xz
        ``(parameters, data, states, grid_data)``. NumPy ``data`` has shape
        ``(n_states, n_x, n_z, n_variables)``; grid points are static metres.
    title
        Title or callback ``title(si, state)``; ``None`` labels state and plane.
    add_bar
        Add a color bar; later animated frames omit it only with both limits set.
    quiver_n
        Draw a wind vector every nth grid point; ``None`` omits vectors.
    quiver_pars
        Overrides for Matplotlib quiver options; not mutated.
    animated
        Mark artists for animation without deriving global color limits.
    rotor_color
        Rotor outline color, or ``None`` to omit overlays. Only turbine centres
        on the exact y slice are drawn.
    kwargs
        Options for ``get_fig``, including caller-owned ``fig`` and ``ax``.

    Yields
    ------
    frame
        Figure or tuple selected by ``get_fig`` return flags. ``nofig`` first
        yields ``None`` without terminating the sequence.

    Raises
    ------
    NotImplementedError
        If the data format is not ``"numpy"``.
    """
    yield from _gen_states_fig(
        fres,
        output,
        states_data_xz,
        "xz",
        title,
        add_bar,
        quiver_n,
        quiver_pars,
        animated,
        rotor_color,
        kwargs,
    )


def _gen_states_fig_yz(
    fres: Any,
    output: Any,
    states_data_yz: tuple[Any, Any, Any, Any],
    title: Any = None,
    add_bar: bool = True,
    quiver_n: int | None = None,
    quiver_pars: dict[str, Any] | None = None,
    animated: bool = False,
    rotor_color: Any = None,
    **kwargs: Any,
) -> Iterator[Any]:
    """Generate vertical yz flow frames with the horizontal axis inverted.

    Parameters
    ----------
    fres
        Read-only farm fields with dimensions ``(FC.STATE, FC.TURBINE)``.
    output
        Executing output supplying ``nofig``.
    states_data_yz
        ``(parameters, data, states, grid_data)``. NumPy ``data`` has shape
        ``(n_states, n_y, n_z, n_variables)``; grid points are static metres.
    title
        Title or callback ``title(si, state)``; ``None`` labels state and plane.
    add_bar
        Add a color bar; later animated frames omit it only with both limits set.
    quiver_n
        Draw a wind vector every nth grid point; ``None`` omits vectors.
    quiver_pars
        Overrides for Matplotlib quiver options; not mutated.
    animated
        Mark artists for animation without deriving global color limits.
    rotor_color
        Rotor outline color, or ``None`` to omit overlays. Only turbine centres
        on the exact x slice are drawn.
    kwargs
        Options for ``get_fig``, including caller-owned ``fig`` and ``ax``.

    Yields
    ------
    frame
        Figure or tuple selected by ``get_fig`` return flags. ``nofig`` first
        yields ``None`` without terminating the sequence.

    Raises
    ------
    NotImplementedError
        If the data format is not ``"numpy"``.
    """
    yield from _gen_states_fig(
        fres,
        output,
        states_data_yz,
        "yz",
        title,
        add_bar,
        quiver_n,
        quiver_pars,
        animated,
        rotor_color,
        kwargs,
    )
