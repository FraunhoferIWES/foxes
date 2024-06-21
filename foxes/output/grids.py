import numpy as np
import pandas as pd
from xarray import Dataset

from foxes.utils import wd2uv, write_nc
import foxes.variables as FV
import foxes.constants as FC


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
        The x grid positions, shape: (n_x,)
    y_pos: numpy.ndarray
        The y grid positions, shape: (n_y,)
    z_pos: float
        The z position of the grid
    g_pts: numpy.ndarray
        The grid points, shape: (n_states, n_pts, 3)

    """

    # prepare:
    n_states = farm_results[FV.H].shape[0]

    # get base rectangle:
    x_min = xmin if xmin is not None else farm_results[FV.X].min().to_numpy() - xspace
    y_min = ymin if ymin is not None else farm_results[FV.Y].min().to_numpy() - yspace
    z_min = z if z is not None else farm_results[FV.H].min().to_numpy()
    x_max = xmax if xmax is not None else farm_results[FV.X].max().to_numpy() + xspace
    y_max = ymax if ymax is not None else farm_results[FV.Y].max().to_numpy() + yspace
    z_max = z if z is not None else farm_results[FV.H].max().to_numpy()

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
        The x grid positions, shape: (n_x,)
    y_pos: float
        The y position of the grid
    z_pos: numpy.ndarray
        The z grid positions, shape: (n_z,)
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
        The y grid positions, shape: (n_y,)
    z_pos: numpy.ndarray
        The z grid positions, shape: (n_z,)
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


def np2np_p(data, a_pos, b_pos):
    """
    Create numpy data from numpy data

    Parameters
    ----------
    data: dict
        The data on the grid. Key: variable name,
        value: numpy.ndarray with shape (n_gpts,)
    a_pos: numpy.ndarray
        The first axis coordinates, e.g. x_pos,
        shape: (n_a,)
    b_pos: numpy.ndarray
        The second axis coordinates, e.g. y_pos,
        shape: (n_b,)

    Returns
    -------
    out: dict
        The data on the grid. Key: variable name,
        value: numpy.ndarray with shape (n_a, n_b, n_vars)

    """
    n_a = len(a_pos)
    n_b = len(b_pos)
    n_v = len(data)
    out = np.zeros((n_a, n_b, n_v), dtype=FC.DTYPE)
    for vi, (v, d) in enumerate(data.items()):
        out[:, :, vi] = d.reshape(n_a, n_b)
    return out


def np2np_sp(data, states, a_pos, b_pos):
    """
    Create numpy data from numpy data

    Parameters
    ----------
    data: dict
        The data on the grid. Key: variable name,
        value: numpy.ndarray with shape (n_states, n_gpts)
    states: numpy.ndarray
        The states index, shape: (n_states,)
    a_pos: numpy.ndarray
        The first axis coordinates, e.g. x_pos,
        shape: (n_a,)
    b_pos: numpy.ndarray
        The second axis coordinates, e.g. y_pos,
        shape: (n_b,)

    Returns
    -------
    out: dict
        The data on the grid. Key: variable name,
        value: numpy.ndarray with shape (n_states, n_a, n_b, n_vars)

    """
    n_s = len(states)
    n_a = len(a_pos)
    n_b = len(b_pos)
    n_v = len(data)
    out = np.zeros((n_s, n_a, n_b, n_v), dtype=FC.DTYPE)
    for vi, (v, d) in enumerate(data.items()):
        out[:, :, :, vi] = d.reshape(n_s, n_a, n_b)
    return out


def np2pd_p(data, a_pos, b_pos, ori, label_map={}):
    """
    Create pandas DataFrame from numpy data

    Parameters
    ----------
    data: dict
        The data on the grid. Key: variable name,
        value: numpy.ndarray with shape (n_gpts,)
    a_pos: numpy.ndarray
        The first axis coordinates, e.g. x_pos,
        shape: (n_a,)
    b_pos: numpy.ndarray
        The second axis coordinates, e.g. y_pos,
        shape: (n_b,)
    ori: str
        The orientation, 'xy' or 'xz' or 'yz'
    label_map: dict
        The mapping from original to new field names

    Returns
    -------
    out: pandas.DataFrame
        The multi-indexed DataFrame object

    """
    a, b = [label_map.get(o, o) for o in ori]
    n_a = len(a_pos)
    n_b = len(b_pos)
    minds = pd.MultiIndex.from_product([range(n_a), range(n_b)], names=[a, b])
    return pd.DataFrame(index=minds, data=data)


def np2pd_sp(data, states, a_pos, b_pos, ori, label_map={}):
    """
    Create pandas DataFrame from numpy data

    Parameters
    ----------
    data: dict
        The data on the grid. Key: variable name,
        value: numpy.ndarray with shape (n_states, n_gpts,)
    states: numpy.ndarray
        The states index, shape: (n_states,)
    a_pos: numpy.ndarray
        The first axis coordinates, e.g. x_pos,
        shape: (n_a,)
    b_pos: numpy.ndarray
        The second axis coordinates, e.g. y_pos,
        shape: (n_b,)
    ori: str
        The orientation, 'xy' or 'xz' or 'yz'
    label_map: dict
        The mapping from original to new field names

    Returns
    -------
    out: pandas.DataFrame
        The multi-indexed DataFrame object

    """
    a, b = [label_map.get(o, o) for o in ori]
    s = label_map.get(FC.STATE, FC.STATE)
    n_a = len(a_pos)
    n_b = len(b_pos)
    minds = pd.MultiIndex.from_product(
        [states, range(n_a), range(n_b)], names=[s, a, b]
    )
    return pd.DataFrame(index=minds, data=data)


def np2xr_p(data, a_pos, b_pos, c_pos, ori, label_map={}):
    """
    Create xarray Dataset from numpy data

    Parameters
    ----------
    data: dict
        The data on the grid. Key: variable name,
        value: numpy.ndarray with shape (n_gpts,)
    a_pos: numpy.ndarray
        The first axis coordinates, e.g. x_pos,
        shape: (n_a,)
    b_pos: numpy.ndarray
        The second axis coordinates, e.g. y_pos,
        shape: (n_b,)
    ori: str
        The orientation, 'xy' or 'xz' or 'yz'
    label_map: dict
        The mapping from original to new field names

    Returns
    -------
    out: xarray.Dataset
        The Dataset object

    """
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


def np2xr_sp(data, states, a_pos, b_pos, c_pos, ori, label_map={}):
    """
    Create xarray Dataset from numpy data

    Parameters
    ----------
    data: dict
        The data on the grid. Key: variable name,
        value: numpy.ndarray with shape (n_states, n_gpts,)
    states: numpy.ndarray
        The states index, shape: (n_states,)
    a_pos: numpy.ndarray
        The first axis coordinates, e.g. x_pos,
        shape: (n_a,)
    b_pos: numpy.ndarray
        The second axis coordinates, e.g. y_pos,
        shape: (n_b,)
    ori: str
        The orientation, 'xy' or 'xz' or 'yz'
    label_map: dict
        The mapping from original to new field names

    Returns
    -------
    out: xarray.Dataset
        The Dataset object

    """
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
            v: ((s, b, a), np.swapaxes(d.reshape(n_s, n_a, n_b), 1, 2))
            for v, d in data.items()
        },
        attrs={c: float(c_pos)},
    )


def data2xr(
    x_pos,
    y_pos,
    z_pos,
    point_results,
    vars=None,
    state_mean=False,
    to_file=None,
    **kwargs,
):
    """
    Converts the image data to xarray data

    Parameter
    ---------
    x_pos: numpy.ndarray or float
        The x grid positions, shape: (n_x, 3)
    y_pos: numpy.ndarray or float
        The y grid positions, shape: (n_y, 3)
    z_pos: numpy.ndarray or float
        The z grid positions, shape: (n_z, 3)
    point_results: xarray.Dataset
        Results of calc_points
    vars: list of str, optional
        Variable selection, or None for all
    state_mean: numpy.ndarray or bool
        Computes mean over states, optionally with
        given weights
    round: dict, optional
        Round variables to given digits, or 'auto'
        for default
    to_file: str, optional
        Write to nc file
    kwargs: dict, optional
        Additional parameters for write_nc

    Returns
    -------
    ds: xarray.Dataset
        The xarray data object

    """
    if vars is None:
        vars = list(point_results.data_vars.keys())
    data = {}
    for v in vars:
        if isinstance(state_mean, np.ndarray):
            data[v] = np.einsum("sp,s->p", point_results[v].to_numpy(), state_mean)
        elif state_mean:
            data[v] = np.mean(point_results[v].to_numpy(), axis=0)
        else:
            data[v] = point_results[v].to_numpy()

    allc = [x_pos, y_pos, z_pos]
    allcn = ["x", "y", "z"]
    ci = [i for i, x in enumerate(allc) if isinstance(x, np.ndarray)]
    cj = [i for i in range(3) if i not in ci][0]
    cl = [len(allc[i]) for i in ci]
    cn = list(reversed([allcn[i] for i in ci]))

    coords = {}
    attrs = {allcn[cj]: allc[cj].to_numpy()}
    if (
        FC.STATE in point_results.coords
        and isinstance(state_mean, bool)
        and not state_mean
    ):
        if point_results.sizes[FC.STATE] > 1:
            coords[FC.STATE] = point_results[FC.STATE].to_numpy()
        else:
            attrs[FC.STATE] = str(point_results[FC.STATE][0].to_numpy())
    coords.update({allcn[i]: allc[i] for i in reversed(ci)})

    dvars = {}
    for v, d in data.items():
        if len(d.shape) == 1:
            dvars[v] = (cn, np.swapaxes(d.reshape(*cl), 0, 1))
        else:
            dvars[v] = ([FC.STATE] + cn, np.swapaxes(d.reshape(d.shape[0], *cl), 1, 2))

    ds = Dataset(coords=coords, data_vars=dvars, attrs=attrs)

    if to_file is not None:
        write_nc(ds, to_file, **kwargs)

    return ds
