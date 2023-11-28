import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from foxes.utils import wd2uv
import foxes.variables as FV
import foxes.constants as FC


def get_fig(
    var,
    fig,
    figsize,
    ax,
    data,
    si,
    s,
    normalize_var,
    levels,
    x_pos,
    y_pos,
    vmin,
    vmax,
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
    normalize_var: float, optional
        Divide the variable by this value
    levels: int
        The number of levels for the contourf plot,
        or None for non-contour image
    x_pos: numpy.ndarray
        The grid x positions, shape: (n_x, 3)
    y_pos: numpy.ndarray
        The grid y positions, shape: (n_y, 3)
    vmin: float
        Minimum variable value
    vmax: float
        Maximum variable value
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
    if normalize_var is not None:
        zz /= normalize_var

    # raw data image:
    if levels is None:
        im = hax.pcolormesh(
            x_pos,
            y_pos,
            zz,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            cmap=cmap,
            animated=animated,
        )

    # contour plot:
    else:
        if vmax is not None and vmin is not None and not isinstance(levels, list):
            lvls = np.linspace(vmin, vmax, levels + 1)
        else:
            lvls = levels
        im = hax.contourf(
            x_pos,
            y_pos,
            zz,
            levels=lvls,
            vmax=vmax,
            vmin=vmin,
            cmap=cmap,
            # animated=animated,
        )

    qv = None
    if quiv is not None and quiv[0] is not None:
        n, pars, wd, ws = quiv
        uv = wd2uv(wd[si], ws[si])
        u = uv[:, 0].reshape([N_x, N_y]).T[::n, ::n]
        v = uv[:, 1].reshape([N_x, N_y]).T[::n, ::n]
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


def calc_point_results(
    algo,
    g_pts,
    farm_results=None,
    seq_iter=None,
    runner=None,
    verbosity=0,
    **kwargs,
):
    """

    Helper function that calculates results at grid points.

    Parameters
    ----------
    algo: foxes.Algorithm
        The algorithm for point calculation
    g_pts: numpy.ndarray
        The grid points, shape: (n_states, n_x, n_y, 3)
    farm_results: xarray.Dataset, optional
        The farm results
    seq_iter: foxes.algorithms.sequential.SequentialIter, optional
        The sequential itarator
    runner: foxes.utils.runners.Runner, optional
        The runner
    verbosity: int
        The verbosity level, 0 = silent
    kwargs: dict, optional
        Additional parameters for algo.calc_points

    """
    averb = None if verbosity == algo.verbosity else algo.verbosity
    if averb is not None:
        algo.verbosity = verbosity
    fres = farm_results if seq_iter is None else seq_iter.farm_results
    if runner is None:
        point_results = algo.calc_points(fres, points=g_pts, **kwargs)
    else:
        kwargs["points"] = g_pts
        point_results = runner.run(algo.calc_points, args=(fres,), kwargs=kwargs)
    if averb is not None:
        algo.verbosity = averb

    return point_results


def get_grid_xy(
    farm_results,
    resolution,
    xmin=None,
    ymin=None,
    xmax=None,
    ymax=None,
    z=None,
    xspace=500.0,
    yspace=500.0,
    verbosity=0,
):
    """
    Helper function that generates 2D grid in a horizontal xy-plane.

    Parameters
    ----------
    farm_results: xarray.Dataset
        The farm results. The calculated variables have
        dimensions (state, turbine)
    resolution: float
        The resolution in m
    xmin: float
        The min x coordinate, or None for automatic
    ymin: float
        The min y coordinate, or None for automatic
    xmax: float
        The max x coordinate, or None for automatic
    ymax: float
        The max y coordinate, or None for automatic
    z: float
        The z coordinate of the plane
    xspace: float
        The extra space in x direction, before and after wind farm
    yspace: float
        The extra space in y direction, before and after wind farm
    verbosity: int, optional
        The verbosity level

    Returns
    -------
    x_pos: numpy.ndarray
        The x grid positions, shape: (n_x, 3)
    y_pos: numpy.ndarray
        The y grid positions, shape: (n_y, 3)
    z_pos: float
        The z position of the grid
    g_pts: numpy.ndarray
        The grid points, shape: (n_states, n_pts, 3)

    """

    # prepare:
    n_states = farm_results[FV.H].shape[0]

    # get base rectangle:
    x_min = xmin if xmin is not None else farm_results[FV.X].min() - xspace
    y_min = ymin if ymin is not None else farm_results[FV.Y].min() - yspace
    z_min = z if z is not None else farm_results[FV.H].min()
    x_max = xmax if xmax is not None else farm_results[FV.X].max() + xspace
    y_max = ymax if ymax is not None else farm_results[FV.Y].max() + yspace
    z_max = z if z is not None else farm_results[FV.H].max()

    x_pos, x_res = np.linspace(
        x_min,
        x_max,
        num=int((x_max - x_min) / resolution) + 1,
        endpoint=True,
        retstep=True,
        dtype=None,
    )
    y_pos, y_res = np.linspace(
        y_min,
        y_max,
        num=int((y_max - y_min) / resolution) + 1,
        endpoint=True,
        retstep=True,
        dtype=None,
    )
    N_x, N_y = len(x_pos), len(y_pos)
    n_pts = len(x_pos) * len(y_pos)
    z_pos = 0.5 * (z_min + z_max)
    g_pts = np.zeros((n_states, N_x, N_y, 3), dtype=FC.DTYPE)
    g_pts[:, :, :, 0] = x_pos[None, :, None]
    g_pts[:, :, :, 1] = y_pos[None, None, :]
    g_pts[:, :, :, 2] = z_pos

    if verbosity > 1:
        print("\nFlowPlots2D plot grid:")
        print("Min XYZ  =", x_min, y_min, z_min)
        print("Max XYZ  =", x_max, y_max, z_max)
        print("Pos Z    =", z_pos)
        print("Res XY   =", x_res, y_res)
        print("Dim XY   =", N_x, N_y)
        print("Grid pts =", n_pts)

    return (
        x_pos,
        y_pos,
        z_pos,
        g_pts.reshape(n_states, n_pts, 3),
    )


def get_grid_xz(
    farm_results,
    resolution,
    x_direction=270,
    xmin=None,
    zmin=0.0,
    xmax=None,
    zmax=None,
    y=None,
    xspace=500.0,
    zspace=500.0,
    verbosity=0,
):
    """
    Helper function that generates 2D grid in a vertical xz-plane.

    Parameters
    ----------
    farm_results: xarray.Dataset
        The farm results. The calculated variables have
        dimensions (state, turbine)
    resolution: float
        The resolution in m
    x_direction: float
        The direction of the x axis, 0 = north
    xmin: float
        The min x coordinate, or None for automatic
    zmin: float
        The min z coordinate
    xmax: float
        The max x coordinate, or None for automatic
    zmax: float
        The max z coordinate, or None for automatic
    y: float
        The y coordinate of the plane
    xspace: float
        The extra space in x direction, before and after wind farm
    zspace: float
        The extra space in z direction, below and above wind farm
    verbosity: int, optional
        The verbosity level

    Returns
    -------
    x_pos: numpy.ndarray
        The x grid positions, shape: (n_x, 3)
    y_pos: float
        The y position of the grid
    z_pos: numpy.ndarray
        The z grid positions, shape: (n_z, 3)
    g_pts: numpy.ndarray
        The grid points, shape: (n_states, n_pts, 3)

    """

    # prepare:
    n_states, n_turbines = farm_results[FV.H].shape
    n_x = np.append(wd2uv(x_direction), [0.0], axis=0)
    n_z = np.array([0.0, 0.0, 1.0])
    n_y = np.cross(n_z, n_x)

    # project to axes:
    xyz = np.zeros((n_states, n_turbines, 3), dtype=FC.DTYPE)
    xyz[:, :, 0] = farm_results[FV.X]
    xyz[:, :, 1] = farm_results[FV.Y]
    xyz[:, :, 2] = farm_results[FV.H]
    xx = np.einsum("std,d->st", xyz, n_x)
    yy = np.einsum("std,d->st", xyz, n_y)
    zz = np.einsum("std,d->st", xyz, n_z)
    del xyz

    # get base rectangle:
    x_min = xmin if xmin is not None else np.min(xx) - xspace
    z_min = zmin if zmin is not None else np.minimum(np.min(zz) - zspace, 0.0)
    y_min = y if y is not None else np.min(yy)
    x_max = xmax if xmax is not None else np.max(xx) + xspace
    z_max = zmax if zmax is not None else np.max(zz) + zspace
    y_max = y if y is not None else np.max(yy)
    del xx, yy, zz

    x_pos, x_res = np.linspace(
        x_min,
        x_max,
        num=int((x_max - x_min) / resolution) + 1,
        endpoint=True,
        retstep=True,
        dtype=None,
    )
    z_pos, z_res = np.linspace(
        z_min,
        z_max,
        num=int((z_max - z_min) / resolution) + 1,
        endpoint=True,
        retstep=True,
        dtype=None,
    )
    N_x, N_z = len(x_pos), len(z_pos)
    n_pts = len(x_pos) * len(z_pos)
    y_pos = 0.5 * (y_min + y_max)
    g_pts = np.zeros((n_states, N_x, N_z, 3), dtype=FC.DTYPE)
    g_pts[:] += x_pos[None, :, None, None] * n_x[None, None, None, :]
    g_pts[:] += y_pos * n_y[None, None, None, :]
    g_pts[:] += z_pos[None, None, :, None] * n_z[None, None, None, :]

    if verbosity > 1:
        print("\nFlowPlots2D plot grid:")
        print("Min XYZ  =", x_min, y_min, z_min)
        print("Max XYZ  =", x_max, y_max, z_max)
        print("Pos Y    =", y_pos)
        print("Res XZ   =", x_res, z_res)
        print("Dim XZ   =", N_x, N_z)
        print("Grid pts =", n_pts)

    return (
        x_pos,
        y_pos,
        z_pos,
        g_pts.reshape(n_states, n_pts, 3),
    )


def get_grid_yz(
    farm_results,
    resolution,
    x_direction=270,
    ymin=None,
    zmin=0.0,
    ymax=None,
    zmax=None,
    x=None,
    yspace=500.0,
    zspace=500.0,
    verbosity=0,
):
    """
    Helper function that generates 2D grid in a vertical yz-plane.

    Parameters
    ----------
    farm_results: xarray.Dataset
        The farm results. The calculated variables have
        dimensions (state, turbine)
    resolution: float
        The resolution in m
    x_direction: float
        The direction of the x axis, 0 = north
    ymin: float
        The min y coordinate, or None for automatic
    zmin: float
        The min z coordinate
    ymax: float
        The max y coordinate, or None for automatic
    zmax: float
        The max z coordinate, or None for automatic
    x: float
        The x coordinate of the plane
    yspace: float
        The extra space in y direction, before and after wind farm
    zspace: float
        The extra space in z direction, below and above wind farm
    verbosity: int, optional
        The verbosity level

    Returns
    -------
    x_pos: float
        The x position of the grid
    y_pos: numpy.ndarray
        The y grid positions, shape: (n_y, 3)
    z_pos: numpy.ndarray
        The z grid positions, shape: (n_z, 3)
    g_pts: numpy.ndarray
        The grid points, shape: (n_states, n_pts, 3)

    """

    # prepare:
    n_states, n_turbines = farm_results[FV.H].shape
    n_x = np.append(wd2uv(x_direction), [0.0], axis=0)
    n_z = np.array([0.0, 0.0, 1.0])
    n_y = np.cross(n_z, n_x)

    # project to axes:
    xyz = np.zeros((n_states, n_turbines, 3), dtype=FC.DTYPE)
    xyz[:, :, 0] = farm_results[FV.X]
    xyz[:, :, 1] = farm_results[FV.Y]
    xyz[:, :, 2] = farm_results[FV.H]
    xx = np.einsum("std,d->st", xyz, n_x)
    yy = np.einsum("std,d->st", xyz, n_y)
    zz = np.einsum("std,d->st", xyz, n_z)
    del xyz

    # get base rectangle:
    y_min = ymin if ymin is not None else np.min(yy) - yspace
    z_min = zmin if zmin is not None else np.minimum(np.min(zz) - zspace, 0.0)
    x_min = x if x is not None else np.min(xx)
    y_max = ymax if ymax is not None else np.max(yy) + yspace
    z_max = zmax if zmax is not None else np.max(zz) + zspace
    x_max = x if x is not None else np.max(xx)
    del xx, yy, zz

    y_pos, y_res = np.linspace(
        y_min,
        y_max,
        num=int((y_max - y_min) / resolution) + 1,
        endpoint=True,
        retstep=True,
        dtype=None,
    )
    z_pos, z_res = np.linspace(
        z_min,
        z_max,
        num=int((z_max - z_min) / resolution) + 1,
        endpoint=True,
        retstep=True,
        dtype=None,
    )
    N_y, N_z = len(y_pos), len(z_pos)
    n_pts = len(y_pos) * len(z_pos)
    x_pos = 0.5 * (x_min + x_max)
    g_pts = np.zeros((n_states, N_y, N_z, 3), dtype=FC.DTYPE)
    g_pts[:] += x_pos * n_x[None, None, None, :]
    g_pts[:] += y_pos[None, :, None, None] * n_y[None, None, None, :]
    g_pts[:] += z_pos[None, None, :, None] * n_z[None, None, None, :]

    if verbosity > 1:
        print("\nFlowPlots2D plot grid:")
        print("Min XYZ  =", x_min, y_min, z_min)
        print("Max XYZ  =", x_max, y_max, z_max)
        print("Pos X    =", x_pos)
        print("Res YZ   =", y_res, z_res)
        print("Dim YZ   =", N_y, N_z)
        print("Grid pts =", n_pts)

    return (
        x_pos,
        y_pos,
        z_pos,
        g_pts.reshape(n_states, n_pts, 3),
    )
