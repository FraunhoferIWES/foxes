import numpy as np

from .stable import logz


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

    :group: utils.abl.unstable

    """
    x = (1.0 - 16.0 * height / mol) ** 0.25
    return (
        2.0 * np.log((1.0 + x) / 2.0)
        + np.log((1.0 + x**2) / 2.0)
        - 2.0 * np.arctan(x)
        + np.pi / 2.0
    )


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

    :group: utils.abl.unstable

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

    :group: utils.abl.unstable

    """
    return ustar / kappa * (logz(height, z0) - psi)
