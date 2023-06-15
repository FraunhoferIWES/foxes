import numpy as np


def wd2wdvec(wd, ws=1.0, axis=-1):
    """
    Calculate wind direction vectors from wind directions
    in degrees.

    Parameters
    ----------
    wd: numpy.ndarray
        Wind direction array (any shape)
    ws: float or numpy.ndarray
        The wind speed. Has to broadcast against wd.
    axis: int
        Location where to insert the (x, y) dimension
        into the shape of wd

    Returns
    -------
    wdvec: numpy.ndarray
        The wind direction vectors

    :group: utils

    """

    wdr = wd * np.pi / 180.0
    n = np.stack([np.sin(wdr), np.cos(wdr)], axis=axis)

    if np.isscalar(ws):
        return ws * n

    return np.expand_dims(ws, axis) * n


def wd2uv(wd, ws=1.0, axis=-1):
    """
    Calculate wind vectors from wind directions
    in degrees.

    Parameters
    ----------
    wd: numpy.ndarray
        Wind direction array (any shape)
    ws: float or numpy.ndarray
        The wind speed. Has to broadcast against wd.
    axis: int
        Axis location where to insert the (u, v) components
        into the shape of wd

    Returns
    -------
    uv: numpy.ndarray
        The wind vectors

    :group: utils

    """
    return -wd2wdvec(wd, ws, axis)


def uv2wd(uv, axis=-1):
    """
    Calculate wind direction from wind vectors.

    Parameters
    ----------
    uv: numpy.ndarray
        The wind vectors, any shape
    axis: int
        The axis which corresponds to (u, v) components

    Returns
    -------
    wd: numpy.ndarray
        The wind direction array

    :group: utils

    """

    if axis == -1:
        u = uv[..., 0]
        v = uv[..., 1]
    else:
        s = tuple(0 if a == axis else slice(None) for a in range(len(uv.shape)))
        u = uv[s]
        s = tuple(1 if a == axis else slice(None) for a in range(len(uv.shape)))
        v = uv[s]

    return np.mod(180 + np.rad2deg(np.arctan2(u, v)), 360)


def wdvec2wd(wdvec, axis=-1):
    """
    Calculate wind direction from wind direction vectors.

    Parameters
    ----------
    wdvec: numpy.ndarray
        The wind direction vectors, any shape
    axis: int
        The axis which corresponds to (x, y) components

    Returns
    -------
    wd: numpy.ndarray
        The wind direction array

    :group: utils

    """
    return uv2wd(-wdvec, axis)


def delta_wd(wd_a, wd_b):
    """
    Calculates wd_b - wd_a.

    Parameters
    ----------
    wd_a: numpy.ndarray
        Array of wind directions.
        Shape: any shape
    wd_b: numpy.ndarray
        Array of wind directions.
        Shape: same as wd_a

    Returns
    -------
    numpy.ndarray :
        Array of wind direction deltas.
        Shape: same as wd_a, wd_b

    :group: utils

    """
    out = wd_b - wd_a

    out[out < -180.0] += 360.0
    out[out > 180.0] -= 360.0

    return out
