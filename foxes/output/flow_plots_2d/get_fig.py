import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse

from foxes.utils import wd2uv


def _draw_rotor_overlay(
    hax,
    show_rotor_dict,
    xlabel,
    ylabel,
    rotor_plane,
    rotor_slice,
    animated,
):
    """
    Draw rotor overlay markers and return created matplotlib artists.
    """
    imr = []
    data_dict = show_rotor_dict

    # Optionally filter turbines to the currently plotted x/y slice.
    if rotor_slice is not None:
        sax = rotor_slice.get("axis", None)
        sval = rotor_slice.get("value", None)
        stol = rotor_slice.get("tol", 0.0)

        if sax in ["x", "y"] and sval is not None:
            src = data_dict["X"] if sax == "x" else data_dict["Y"]
            mask = np.isclose(src, sval, atol=stol, rtol=0.0)
            if np.any(mask):
                data_dict = {
                    k: (v[mask] if hasattr(v, "__len__") and len(v) == len(src) else v)
                    for k, v in data_dict.items()
                }
            else:
                return imr

    D = data_dict["D"]
    if len(D) == 0:
        return imr

    if rotor_plane is None:
        if (xlabel == "x [m]") and (ylabel == "y [m]"):
            rotor_plane = "xy"
        elif (xlabel == "x [m]") and (ylabel == "z [m]"):
            rotor_plane = "xz"
        elif (xlabel == "y [m]") and (ylabel == "z [m]"):
            rotor_plane = "yz"

    c = data_dict["color"]

    if rotor_plane == "xy":
        x = data_dict["X"]
        y = data_dict["Y"]
        turb_angle = data_dict["turb_angle"]
        theta = np.deg2rad(np.mod(turb_angle + 90, 360))
        coords = np.zeros(shape=(2, len(D)))
        coords[0, :] = (D / 2) * np.sin(theta)
        coords[1, :] = (D / 2) * np.cos(theta)

        for t in np.arange(len(D)):
            turb_x1 = x[t] + coords[0, t]
            turb_x2 = x[t] - coords[0, t]
            turb_y1 = y[t] + coords[1, t]
            turb_y2 = y[t] - coords[1, t]
            imr += hax.plot(
                [turb_x1, turb_x2],
                [turb_y1, turb_y2],
                color=c,
                linestyle="-",
                linewidth=1,
                animated=animated,
            )

        return imr

    for t in np.arange(len(D)):
        turb_angle = data_dict["turb_angle"][t]
        theta = np.deg2rad(np.mod(turb_angle + 90.0, 360.0))
        n_x = np.cos(theta)
        n_y = -np.sin(theta)
        R = D[t] / 2.0

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
    var,
    data,
    si,
    s,
    x_pos,
    y_pos,
    fig=None,
    figsize=None,
    ax=None,
    levels=None,
    cmap=None,
    xlabel=None,
    ylabel=None,
    title=None,
    add_bar=True,
    vlabel=None,
    ret_state=False,
    ret_im=False,
    vmin=None,
    vmax=None,
    quiv=None,
    invert_axis=None,
    animated=False,
    show_rotor_dict=None,
    rotor_plane=None,
    rotor_slice=None,
):
    """
    Helper function that creates the flow image plot.

    Parameters
    ----------
    var: str
        The variable name
    fig: plt.Figure, optional
        The figure object
    figsize: tuple
        The figsize for plt.Figure
    ax: plt.Axes, optional
        The figure axes
    data: numpy.ndarray
        The grid data to plot, shape: (n_states, n_x, x_y)
    si: int, optional
        The state counter
    s: object
        The state index
    levels: int
        The number of levels for the contourf plot,
        or None for non-contour image
    x_pos: numpy.ndarray
        The grid x positions, shape: (n_x, 3)
    y_pos: numpy.ndarray
        The grid y positions, shape: (n_y, 3)
    xlabel: str, optional
        The x axis label
    ylabel: str, optional
        The y axis label
    title: str, optional
        The title
    add_bar: bool
        Add a color bar
    vlabel: str, optional
        The variable label
    ret_state: bool
        Flag for state index return
    ret_im: bool
        Flag for image return
    vmin: float, optional
        The minimal variable value
    vmax: float, optional
        The maximal variable value
    quiv: tuple, optional
        The quiver data: (n, pars, wd, ws)
    invert_axis: str, optional
        Which axis to invert, either x or y
    animated: bool
        Switch for usage for an animation
    show_rotor_dict: dict, optional
        Parameters for indicating the rotor plane
        by a line in xy or by projected disk markers
        (ellipse/circle, and line for edge-on) in xz/yz
    rotor_plane: str, optional
        The rotor plotting plane, one of xy, xz, yz
    rotor_slice: dict, optional
        Optional slice filter for rotor plotting, with keys:
        axis (x or y), value (float), tol (float)

    Yields
    ------
    fig: matplotlib.Figure
        The figure object
    si: int, optional
        The state index
    im: tuple, optional
        The image objects, matplotlib.collections.QuadMesh
        or matplotlib.QuadContourSet

    """

    # create plot:
    if fig is None:
        hfig = plt.figure(figsize=figsize)
    else:
        hfig = fig
    if ax is None:
        hax = hfig.add_subplot(111)
    else:
        hax = ax

    # get results:
    N_x = len(x_pos)
    N_y = len(y_pos)
    zz = data[si].reshape(N_x, N_y).T

    # raw data image:
    if levels is None:
        im = hax.pcolormesh(
            x_pos,
            y_pos,
            zz,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            animated=animated,
        )

    # contour plot:
    else:
        im = hax.contourf(
            x_pos,
            y_pos,
            zz,
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            # animated=animated,
        )

    qv = None
    if quiv is not None and quiv[0] is not None:
        n, pars, wd, ws = quiv
        uv = wd2uv(wd[si], ws[si])
        u = uv[:, :, 0].T[::n, ::n]
        v = uv[:, :, 1].T[::n, ::n]
        qv = hax.quiver(x_pos[::n], y_pos[::n], u, v, animated=animated, **pars)
        del n, pars, u, v, uv

    if xlabel is None or ylabel is None:
        if rotor_plane == "xz":
            xlabel = "x [m]" if xlabel is None else xlabel
            ylabel = "z [m]" if ylabel is None else ylabel
        elif rotor_plane == "yz":
            xlabel = "y [m]" if xlabel is None else xlabel
            ylabel = "z [m]" if ylabel is None else ylabel
        else:
            xlabel = "x [m]" if xlabel is None else xlabel
            ylabel = "y [m]" if ylabel is None else ylabel

    hax.autoscale_view()
    hax.set_xlabel(xlabel)
    hax.set_ylabel(ylabel)
    hax.set_aspect("equal", adjustable="box")
    hax.set_xlim(x_pos.min(), x_pos.max())
    hax.set_ylim(y_pos.min(), y_pos.max())

    ttl = None
    if animated:
        if title is None:
            if hasattr(s, "dtype") and np.issubdtype(s.dtype, np.datetime64):
                t = np.datetime_as_string(s, unit="m").replace("T", " ")
            else:
                t = f"State {s}"
        else:
            t = title
        ttl = hax.text(
            0.5,
            1.05,
            t,
            backgroundcolor="w",
            transform=hax.transAxes,
            ha="center",
            animated=True,
            clip_on=False,
        )
    else:
        hax.set_title(title if title is not None else f"State {s}")

    if invert_axis == "x":
        hax.invert_xaxis()
    elif invert_axis == "y":
        hax.invert_yaxis()

    # add rotor position:
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
        out = hfig
    else:
        out = fig

    if ret_state or ret_im:
        out = [out]
    if ret_state:
        out.append(si)
    if ret_im:
        out.append([i for i in [im, qv, ttl] if i is not None] + imr)
    if ret_state or ret_im:
        out = tuple(out)

    return out
