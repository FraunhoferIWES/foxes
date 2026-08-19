from typing import cast

import numpy as np

import foxes.constants as FC


def tke2ti(tke: np.ndarray, ws: np.ndarray, max_ti: float | None = None) -> np.ndarray:
    """
    Convert turbulent kinetic energy (TKE) to turbulence intensity (TI).

    Parameters
    ----------
    tke
        Turbulent kinetic energy.
    ws
        Wind speed.
    max_ti
        Upper limit of the computed TI values.

    Returns
    -------
    ti
        Turbulence intensity.


    """
    ti = np.sqrt(1.5 * tke) / ws
    if max_ti is not None:
        ti = np.minimum(ti, max_ti)

    return cast(np.ndarray, ti)


def ustar2ti(
    ustar: np.ndarray, ws: np.ndarray, max_ti: float | None = None
) -> np.ndarray:
    """
    Convert friction velocity (u*) to turbulence intensity (TI).

    Parameters
    ----------
    ustar
        Friction velocity.
    ws
        Wind speed.
    max_ti
        Upper limit of the computed TI values.

    Returns
    -------
    ti
        Turbulence intensity.


    """

    ti = (ustar / FC.KAPPA) / ws
    if max_ti is not None:
        ti = np.minimum(ti, max_ti)

    return cast(np.ndarray, ti)
