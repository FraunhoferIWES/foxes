import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from foxes.utils import wd2uv

def get_fig(
    var,
    fig,
    figsize,
    ax,
    data,
    si,
    s,
    levels,
    x_pos,
    y_pos,
    cmap,
    xlabel,
    ylabel,
    title,
    add_bar,
    vlabel,
    ret_state,
    ret_im,
    quiv=None,
    invert_axis=None,
    animated=False,
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
    xlabel: str
        The x axis label
    ylabel: str
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
    quiv: tuple, optional
        The quiver data: (n, pars, wd, ws)
    invert_axis: str, optional
        Which axis to invert, either x or y
    animated: bool
        Switch for usage for an animation

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

    hax.autoscale_view()
    hax.set_xlabel(xlabel)
    hax.set_ylabel(ylabel)
    hax.set_aspect("equal", adjustable="box")

    ttl = None
    if animated and title is None:
        if hasattr(s, "dtype") and np.issubdtype(s.dtype, np.datetime64):
            t = np.datetime_as_string(s, unit="m").replace("T", " ")
        else:
            t = s
        ttl = hax.text(
            0.5,
            1.05,
            f"State {t}",
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
        out.append([i for i in [im, qv, ttl] if i is not None])
    if ret_state or ret_im:
        out = tuple(out)

    return out


