from typing import cast

import numpy as np

from .stable import logz


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


    """
    x = (1.0 - 16.0 * height / mol) ** 0.25
    return cast(
        float | np.ndarray,
        2.0 * np.log((1.0 + x) / 2.0)
        + np.log((1.0 + x**2) / 2.0)
        - 2.0 * np.arctan(x)
        + np.pi / 2.0,
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


    """
    return cast(float | np.ndarray, ustar / kappa * (logz(height, z0) - psi))
