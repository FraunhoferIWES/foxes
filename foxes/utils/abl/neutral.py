import numpy as np


def logz(height: float | np.ndarray, z0: float | np.ndarray) -> float | np.ndarray:
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


def ustar(
    ws_ref: float | np.ndarray,
    h_ref: float | np.ndarray,
    z0: float | np.ndarray,
    kappa: float = 0.41,
) -> float | np.ndarray:
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
        The von Karman constant

    Returns
    -------
    ustar: float
        The friction velocity

    :group: utils.abl.neutral

    """
    lz = logz(h_ref, z0)
    return ws_ref * kappa / lz


def calc_ws(
    height: float | np.ndarray,
    z0: float | np.ndarray,
    ustar: float | np.ndarray,
    kappa: float = 0.41,
) -> float | np.ndarray:
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
        The von Karman constant

    Returns
    -------
    ws: float
        The wind speed

    :group: utils.abl.neutral

    """
    return ustar / kappa * logz(height, z0)
