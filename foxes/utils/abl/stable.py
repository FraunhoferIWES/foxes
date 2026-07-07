import numpy as np
from .neutral import logz as lgz


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

    :group: utils.abl.stable

    """
    return lgz(height, z0)


def psi(height: float | np.ndarray, mol: float | np.ndarray) -> float | np.ndarray:
    """
    The Psi function

    Parameters
    ----------
    height: float
        The height value
    mol: float
        The Monin-Obukhov height

    Returns
    -------
    psi: float
        The Psi function value

    :group: utils.abl.stable

    """
    h = np.minimum(height, np.abs(mol))
    return -5.0 * h / mol


def ustar(
    ws_ref: float | np.ndarray,
    h_ref: float | np.ndarray,
    z0: float | np.ndarray,
    mol: float | np.ndarray,
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
    mol: float
        The Monin-Obukhov height
    kappa: float
        The von Karman constant

    Returns
    -------
    ustar: float
        The friction velocity

    :group: utils.abl.stable

    """
    return ws_ref * kappa / (logz(h_ref, z0) - psi(h_ref, mol))


def calc_ws(
    height: float | np.ndarray,
    z0: float | np.ndarray,
    ustar: float | np.ndarray,
    psi: float | np.ndarray,
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
    psi: float
        The Psi function values
    kappa: float
        The von Karman constant

    Returns
    -------
    ws: float
        The wind speed

    :group: utils.abl.stable

    """
    return ustar / kappa * (logz(height, z0) - psi)
