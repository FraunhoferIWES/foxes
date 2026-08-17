from typing import cast

import numpy as np
from .neutral import logz as lgz


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

    :group: utils.abl.stable

    """
    return cast(float | np.ndarray, lgz(height, z0))


def psi(height: float | np.ndarray, mol: float | np.ndarray) -> float | np.ndarray:
    """
    The Psi function

    Parameters
    ----------
    height
        The height value
    mol
        The Monin-Obukhov height

    Returns
    -------
    psi
        The Psi function value

    :group: utils.abl.stable

    """
    h = np.minimum(height, np.abs(mol))
    return cast(float | np.ndarray, -5.0 * h / mol)


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
    ws_ref
        The reference wind speed
    h_ref
        The reference height
    z0
        The roughness length
    mol
        The Monin-Obukhov height
    kappa
        The von Karman constant

    Returns
    -------
    ustar
        The friction velocity

    :group: utils.abl.stable

    """
    return cast(
        float | np.ndarray, ws_ref * kappa / (logz(h_ref, z0) - psi(h_ref, mol))
    )


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
    height
        The evaluation height
    z0
        The roughness length
    ustar
        The friction velocity
    psi
        The Psi function values
    kappa
        The von Karman constant

    Returns
    -------
    ws
        The wind speed

    :group: utils.abl.stable

    """
    return cast(float | np.ndarray, ustar / kappa * (logz(height, z0) - psi))
