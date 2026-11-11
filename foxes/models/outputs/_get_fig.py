"""Stateless Matplotlib flow-field and rotor-overlay rendering."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable

from foxes.utils import wd2uv


def _draw_rotor_overlay(
    hax: Any,
    show_rotor_dict: dict[str, Any],
    xlabel: str | None,
    ylabel: str | None,
    rotor_plane: str | None,
    rotor_slice: dict[str, Any] | None,
    animated: bool,
) -> list[Any]:
    """Draw projected rotors on the supplied axes without changing source arrays.

    Parameters
    ----------
    hax
        Caller-owned axes receiving the artists.
    show_rotor_dict
        Turbine arrays ``X``, ``Y``, ``H``, ``D`` in metres, ``turb_angle`` in
        degrees, and a Matplotlib ``color``.
    xlabel
        Horizontal label used to infer a plane when ``rotor_plane`` is ``None``.
    ylabel
        Vertical label used to infer a plane when ``rotor_plane`` is ``None``.
    rotor_plane
        ``"xy"``, ``"xz"`` or ``"yz"``; unrecognized planes draw no artists.
    rotor_slice
        Filter with ``axis`` (``"x"`` or ``"y"``), ``value`` and absolute
        ``tol`` in metres. Only intersecting turbine centres are retained.
    animated
        Mark created artists for animation.

    Returns
    -------
    artists
        Added rotor lines or projected ellipses, in turbine order.
    """
    data_dict = _filter_rotors(show_rotor_dict, rotor_slice)
    if data_dict is None or len(data_dict["D"]) == 0:
        return []
    if rotor_plane is None:
        if (xlabel == "x [m]") and (ylabel == "y [m]"):
            rotor_plane = "xy"
        elif (xlabel == "x [m]") and (ylabel == "z [m]"):
            rotor_plane = "xz"
        elif (xlabel == "y [m]") and (ylabel == "z [m]"):
            rotor_plane = "yz"

    if rotor_plane == "xy":
        return _draw_horizontal_rotors(hax, data_dict, animated)
    return _draw_vertical_rotors(hax, data_dict, rotor_plane, animated)


def _filter_rotors(data: dict, rotor_slice: dict | None) -> dict | None:
    """Select turbine centres intersecting a slice without mutating geometry.

    Parameters
    ----------
    data
        Rotor arrays and color consumed by ``_draw_rotor_overlay``.
    rotor_slice
        Axis, value and absolute tolerance in metres, or ``None`` for no filter.

    Returns
    -------
    selected
        The original mapping, a filtered mapping with copied turbine arrays,
        or ``None`` when no turbine centre intersects the slice.
    """
    if rotor_slice is None:
        return data
    axis = rotor_slice.get("axis")
    value = rotor_slice.get("value")
    if axis not in ("x", "y") or value is None:
        return data
    source = data["X"] if axis == "x" else data["Y"]
    mask = np.isclose(source, value, atol=rotor_slice.get("tol", 0.0), rtol=0.0)
    if not np.any(mask):
        return None
    return {
        key: values[mask]
        if hasattr(values, "__len__") and len(values) == len(source)
        else values
        for key, values in data.items()
    }


def _draw_horizontal_rotors(hax: Axes, data: dict, animated: bool) -> list:
    """Add rotor-diameter lines in an xy plane.

    Parameters
    ----------
    hax
        Axes receiving the artists.
    data
        Read-only turbine geometry in metres and yaw angles in degrees.
    animated
        Mark created lines for animation.

    Returns
    -------
    artists
        Added lines in turbine order.
    """
    theta = np.deg2rad(np.mod(data["turb_angle"] + 90, 360))
    coordinates = np.zeros((2, len(data["D"])))
    coordinates[0] = (data["D"] / 2) * np.sin(theta)
    coordinates[1] = (data["D"] / 2) * np.cos(theta)
    artists = []
    for ti in range(len(data["D"])):
        artists += hax.plot(
            [data["X"][ti] + coordinates[0, ti], data["X"][ti] - coordinates[0, ti]],
            [data["Y"][ti] + coordinates[1, ti], data["Y"][ti] - coordinates[1, ti]],
            color=data["color"],
            linestyle="-",
            linewidth=1,
            animated=animated,
        )
    return artists


def _draw_vertical_rotors(
    hax: Axes, data_dict: dict, rotor_plane: str | None, animated: bool
) -> list:
    """Add projected rotor disks, reducing edge-on projections to lines.

    Parameters
    ----------
    hax
        Axes receiving the artists.
    data_dict
        Read-only turbine geometry in metres and yaw angles in degrees.
    rotor_plane
        ``"xz"`` or ``"yz"``; other values draw nothing.
    animated
        Mark created artists for animation.

    Returns
    -------
    artists
        Added ellipses or edge-on lines in turbine order.
    """
    imr: list[Any] = []
    c = data_dict["color"]
    for t in range(len(data_dict["D"])):
        turb_angle = data_dict["turb_angle"][t]
        theta = np.deg2rad(np.mod(turb_angle + 90.0, 360.0))
        n_x = np.cos(theta)
        n_y = -np.sin(theta)
        R = data_dict["D"][t] / 2.0

        if rotor_plane == "yz":
            xc = data_dict["Y"][t]
            yc = data_dict["H"][t]
            width = 2.0 * R * abs(n_x)
            height = 2.0 * R
        elif rotor_plane == "xz":
            xc = data_dict["X"][t]
            yc = data_dict["H"][t]
            width = 2.0 * R * abs(n_y)
            height = 2.0 * R
        else:
            continue

        if np.isclose(width, 0.0, atol=1.0e-9 * max(2.0 * R, 1.0), rtol=0.0):
            imr += hax.plot(
                [xc, xc],
                [yc - R, yc + R],
                color=c,
                linestyle="-",
                linewidth=1,
                animated=animated,
            )
        else:
            ep = Ellipse(
                (xc, yc),
                width=width,
                height=height,
                angle=0.0,
                fill=False,
                color=c,
                linewidth=1,
                animated=animated,
            )
            hax.add_patch(ep)
            imr.append(ep)

    return imr


def get_fig(
    var: str,
    data: np.ndarray,
    si: int,
    s: Any,
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    fig: Figure | None = None,
    figsize: tuple[int, int] | None = None,
    ax: Axes | None = None,
    levels: int | None = None,
    cmap: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    add_bar: bool = True,
    vlabel: str | None = None,
    ret_state: bool = False,
    ret_im: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    quiv: Any = None,
    invert_axis: str | None = None,
    animated: bool = False,
    show_rotor_dict: dict[str, Any] | None = None,
    rotor_plane: str | None = None,
    rotor_slice: dict[str, Any] | None = None,
) -> Any:
    """Render one flow-field state on new or caller-supplied axes.

    Parameters
    ----------
    var
        FOXES variable name used as the default color-bar label.
    data
        Read-only grid values with shape ``(n_states, n_x, n_y)`` or
        ``(n_states, n_x * n_y)`` in C ordering.
    si
        Positional state index into ``data`` and quiver arrays.
    s
        State label used for the automatic title.
    x_pos
        Horizontal coordinates in metres, shape ``(n_x,)``.
    y_pos
        Vertical coordinates in metres, shape ``(n_y,)``.
    fig
        Figure receiving a new subplot when ``ax`` is ``None``.
    figsize
        Figure size in inches when creating a figure.
    ax
        Caller-owned axes to draw on without replacing or clearing them.
    levels
        Contour levels; ``None`` selects a pseudocolor mesh.
    cmap
        Matplotlib colormap name.
    xlabel
        Horizontal label, or ``None`` for the plotting-plane coordinate in metres.
    ylabel
        Vertical label, or ``None`` for the plotting-plane coordinate in metres.
    title
        Explicit title, or ``None`` for a state-derived title.
    add_bar
        Append a vertical color bar to the right of the plotting axes.
    vlabel
        Color-bar label, or ``None`` to use ``var``.
    ret_state
        Include the positional state index in the returned tuple.
    ret_im
        Include the ordered list of created artists in the returned tuple.
    vmin
        Lower color limit, or ``None`` for automatic normalization.
    vmax
        Upper color limit, or ``None`` for automatic normalization.
    quiv
        ``(stride, options, wd, ws)`` with wind arrays shaped
        ``(n_states, n_x, n_y)``; ``None`` omits vectors.
    invert_axis
        ``"x"`` or ``"y"`` inverts that plotted axis; other values do not invert.
    animated
        Mark mesh, quiver, title and rotor artists for animation. Filled contours
        retain their existing Matplotlib animation behavior.
    show_rotor_dict
        Read-only turbine arrays ``X``, ``Y``, ``H``, ``D`` in metres,
        ``turb_angle`` in degrees, and a Matplotlib ``color``. ``None`` omits
        overlays. Rotors are lines in xy and projected disks in vertical planes.
    rotor_plane
        ``"xy"``, ``"xz"`` or ``"yz"``; ``None`` infers rotor projection from
        axis labels and uses xy coordinate-label defaults.
    rotor_slice
        Centre filter with keys ``axis`` (``"x"`` or ``"y"``), ``value`` and
        absolute ``tol`` in metres; ``None`` draws every rotor.

    Returns
    -------
    frame
        Figure when adding a color bar; otherwise the supplied ``fig`` value
        (possibly ``None``). With return flags, a tuple containing that value,
        then ``si`` if ``ret_state``, then the artist list if ``ret_im``.
        Artists are ordered as field, quiver, animated title, then rotors;
        absent artists are omitted. No figure is saved, shown or closed.
    """

    if fig is None:
        hfig = plt.figure(figsize=figsize)
    else:
        hfig = fig
    if ax is None:
        hax = hfig.add_subplot(111)
    else:
        hax = ax

    im = _field_image(hax, data[si], x_pos, y_pos, levels, cmap, vmin, vmax, animated)
    qv = _quiver_image(hax, quiv, si, x_pos, y_pos, animated)
    xlabel, ylabel = _label_axes(hax, x_pos, y_pos, xlabel, ylabel, rotor_plane)
    ttl = _plot_title(hax, s, title, animated)

    if invert_axis == "x":
        hax.invert_xaxis()
    elif invert_axis == "y":
        hax.invert_yaxis()

    imr = []
    if show_rotor_dict is not None:
        imr = _draw_rotor_overlay(
            hax=hax,
            show_rotor_dict=show_rotor_dict,
            xlabel=xlabel,
            ylabel=ylabel,
            rotor_plane=rotor_plane,
            rotor_slice=rotor_slice,
            animated=animated,
        )

    if add_bar:
        divider = make_axes_locatable(hax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        vlab = vlabel if vlabel is not None else var
        hfig.colorbar(im, cax=cax, orientation="vertical", label=vlab)
        out: Any = hfig
    else:
        out = fig

    artists = [artist for artist in [im, qv, ttl] if artist is not None] + imr
    return _frame_result(out, si, artists, ret_state, ret_im)


def _frame_result(
    figure: Figure | None, si: int, artists: list, ret_state: bool, ret_im: bool
) -> Any:
    """Package one frame using the established conditional tuple layout.

    Parameters
    ----------
    figure
        Figure return value, possibly ``None`` without a color bar.
    si
        Positional state index.
    artists
        Ordered field, quiver, title and rotor artists.
    ret_state
        Include the positional state index after the figure.
    ret_im
        Include the artist list after any requested state index.

    Returns
    -------
    frame
        Figure alone when both flags are disabled; otherwise a tuple ordered
        as figure, requested state index, requested artist list.
    """
    if not ret_state and not ret_im:
        return figure
    values: list[Any] = [figure]
    if ret_state:
        values.append(si)
    if ret_im:
        values.append(artists)
    return tuple(values)


def _field_image(
    ax: Axes,
    data: np.ndarray,
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    levels: int | None,
    cmap: str | None,
    vmin: float | None,
    vmax: float | None,
    animated: bool,
) -> Any:
    """Draw a scalar field with the established transpose and contour rules.

    Parameters
    ----------
    ax
        Caller-owned axes receiving the field.
    data
        One state's values, reshaped to ``(n_x, n_y)`` without mutation.
    x_pos
        Horizontal coordinates, shape ``(n_x,)``.
    y_pos
        Vertical coordinates, shape ``(n_y,)``.
    levels
        Contour levels; ``None`` selects a pseudocolor mesh.
    cmap
        Matplotlib colormap name.
    vmin
        Lower color limit, or ``None`` for automatic normalization.
    vmax
        Upper color limit, or ``None`` for automatic normalization.
    animated
        Mark pseudocolor artists for animation; contour artists retain their
        existing Matplotlib behavior.

    Returns
    -------
    image
        Added mesh or filled-contour artist.
    """
    values = data.reshape(len(x_pos), len(y_pos)).T
    if levels is None:
        return ax.pcolormesh(
            x_pos,
            y_pos,
            values,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            animated=animated,
        )
    return ax.contourf(
        x_pos, y_pos, values, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax
    )


def _quiver_image(
    ax: Axes,
    quiv: Any,
    si: int,
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    animated: bool,
) -> Any:
    """Draw subsampled wind vectors without changing their source arrays.

    Parameters
    ----------
    ax
        Caller-owned axes receiving vectors.
    quiv
        ``(stride, options, wd, ws)``, with wind arrays shaped
        ``(n_states, n_x, n_y)``; ``None`` or a ``None`` stride omits vectors.
    si
        Positional state index.
    x_pos
        Horizontal grid coordinates, shape ``(n_x,)``.
    y_pos
        Vertical grid coordinates, shape ``(n_y,)``.
    animated
        Mark vectors for animation.

    Returns
    -------
    vectors
        Added quiver artist, or ``None`` when vectors are disabled.
    """
    if quiv is None or quiv[0] is None:
        return None
    stride, options, direction, speed = quiv
    vectors = wd2uv(direction[si], speed[si])
    return ax.quiver(
        x_pos[::stride],
        y_pos[::stride],
        vectors[:, :, 0].T[::stride, ::stride],
        vectors[:, :, 1].T[::stride, ::stride],
        animated=animated,
        **options,
    )


def _label_axes(
    ax: Axes,
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    xlabel: str | None,
    ylabel: str | None,
    rotor_plane: str | None,
) -> tuple[str, str]:
    """Apply plane-aware coordinate labels, equal aspect and grid bounds.

    Parameters
    ----------
    ax
        Axes to update in place.
    x_pos
        Horizontal grid coordinates in metres.
    y_pos
        Vertical grid coordinates in metres.
    xlabel
        Explicit horizontal label, or ``None`` for the plane default.
    ylabel
        Explicit vertical label, or ``None`` for the plane default.
    rotor_plane
        ``"xz"`` or ``"yz"`` selects vertical-plane labels; other values use xy.

    Returns
    -------
    horizontal_label
        Applied horizontal label.
    vertical_label
        Applied vertical label.
    """
    plane = rotor_plane if rotor_plane in ("xz", "yz") else "xy"
    xlabel = f"{plane[0]} [m]" if xlabel is None else xlabel
    ylabel = f"{plane[1]} [m]" if ylabel is None else ylabel
    ax.autoscale_view()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_pos.min(), x_pos.max())
    ax.set_ylim(y_pos.min(), y_pos.max())
    return xlabel, ylabel


def _plot_title(ax: Axes, state: Any, title: str | None, animated: bool) -> Any:
    """Apply a static title or create a separate animated title artist.

    Parameters
    ----------
    ax
        Caller-owned axes to update.
    state
        State label; animated datetime labels use minute-resolution timestamps.
    title
        Explicit title text, or ``None`` for a state-derived title.
    animated
        Draw separate animated text rather than replacing the axes title.

    Returns
    -------
    title_artist
        Animated text artist, or ``None`` after setting a static title.
    """
    if not animated:
        ax.set_title(title if title is not None else f"State {state}")
        return None
    if title is None:
        if hasattr(state, "dtype") and np.issubdtype(state.dtype, np.datetime64):
            title = np.datetime_as_string(state, unit="m").replace("T", " ")
        else:
            title = f"State {state}"
    return ax.text(
        0.5,
        1.05,
        title,
        backgroundcolor="w",
        transform=ax.transAxes,
        ha="center",
        animated=True,
        clip_on=False,
    )
