import numpy as np


def logz(height, z0):
    """
    Calculates the log factor for
    wind speed profiles.

    Parameters
    ----------
    height: float
        The evaluation height
    z0: float
        The roughness length

    Returns
    -------
    lz: float
        The log factor

    :group: utils.abl.neutral

    """
    h = np.maximum(height, z0)
    return np.log(h / z0)


def ustar(ws_ref, h_ref, z0, kappa=0.41):
    """
    Calculates the friction velocity,
    based on reference data.

    Parameters
    ----------
    ws_ref: float
        The reference wind speed
    h_ref: float
        The reference height
    z0: float
        The roughness length
    kappa: float
        The van-Karman constant

    Returns
    -------
    ustar: float
        The friction velocity

    :group: utils.abl.neutral

    """
    lz = logz(h_ref, z0)
    return ws_ref * kappa / lz


def calc_ws(height, z0, ustar, kappa=0.41):
    """
    Calculate wind speeds at given height

    Parameters
    ----------
    height: float
        The evaluation height
    z0: float
        The roughness length
    ustar: float
        The friction velocity
    kappa: float
        The van-Karman constant

    Returns
    -------
    ws: float
        The wind speed

    :group: utils.abl.neutral

    """
    return ustar / kappa * logz(height, z0)
