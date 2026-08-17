import numpy as np


def wd2wdvec(
    wd: np.ndarray, ws: float | np.ndarray = 1.0, axis: int = -1
) -> np.ndarray:
    """
    Calculate wind direction vectors from wind directions
    in degrees.

    Parameters
    ----------
    wd
        Wind direction array (any shape)
    ws
        The wind speed. Has to broadcast against wd.
    axis
        Location where to insert the (x, y) dimension
        into the shape of wd

    Returns
    -------
    wdvec
        The wind direction vectors

    :group: utils

    """
    wdr = wd * np.pi / 180.0
    n = np.stack([np.sin(wdr), np.cos(wdr)], axis=axis)

    if np.isscalar(ws):
        return np.asarray(ws * n)

    return np.expand_dims(ws, axis) * n


def wd2uv(wd: np.ndarray, ws: float | np.ndarray = 1.0, axis: int = -1) -> np.ndarray:
    """
    Calculate wind vectors from wind directions
    in degrees.

    Parameters
    ----------
    wd
        Wind direction array (any shape)
    ws
        The wind speed. Has to broadcast against wd.
    axis
        Axis location where to insert the (u, v) components
        into the shape of wd

    Returns
    -------
    uv
        The wind vectors

    :group: utils

    """
    return -wd2wdvec(wd, ws, axis)


def uv2wd(uv: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Calculate wind direction from wind vectors.

    Parameters
    ----------
    uv
        The wind vectors, any shape
    axis
        The axis which corresponds to (u, v) components

    Returns
    -------
    wd
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


def wdvec2wd(wdvec: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Calculate wind direction from wind direction vectors.

    Parameters
    ----------
    wdvec
        The wind direction vectors, any shape
    axis
        The axis which corresponds to (x, y) components

    Returns
    -------
    wd
        The wind direction array

    :group: utils

    """
    return uv2wd(-wdvec, axis)


def delta_wd(wd_a: np.ndarray, wd_b: np.ndarray) -> np.ndarray:
    """
    Calculates wd_b - wd_a.

    Parameters
    ----------
    wd_a
        Array of wind directions.
        Shape
    wd_b
        Array of wind directions.
        Shape: same as wd_a

    Returns
    -------
    Array
        Array of wind direction deltas.
        Shape: same as wd_a, wd_b

    :group: utils

    """
    out = wd_b - wd_a

    out[out < -180.0] += 360.0
    out[out > 180.0] -= 360.0

    return out
