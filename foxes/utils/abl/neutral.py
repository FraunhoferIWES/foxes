import numpy as np


def logz(height: float | np.ndarray, z0: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates the log factor for
    wind speed profiles.

    Parameters
    ----------
    height
        The evaluation height
    z0
        The roughness length

    Returns
    -------
    lz
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
    ws_ref
        The reference wind speed
    h_ref
        The reference height
    z0
        The roughness length
    kappa
        The von Karman constant

    Returns
    -------
    ustar
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
    height
        The evaluation height
    z0
        The roughness length
    ustar
        The friction velocity
    kappa
        The von Karman constant

    Returns
    -------
    ws
        The wind speed

    :group: utils.abl.neutral

    """
    return ustar / kappa * logz(height, z0)
