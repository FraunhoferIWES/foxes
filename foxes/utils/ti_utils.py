import numpy as np

import foxes.constants as FC


def tke2ti(tke, ws, max_ti=None):
    """
    Convert turbulent kinetic energy (TKE) to turbulence intensity (TI).

    Parameters
    ----------
    tke : numpy.ndarray
        Turbulent kinetic energy.
    ws : numpy.ndarray
        Wind speed.
    max_ti : float, optional
        Upper limit of the computed TI values.

    Returns
    -------
    ti :numpy.ndarray
        Turbulence intensity.

    :group: utils

    """
    ti = np.sqrt(1.5 * tke) / ws
    if max_ti is not None:
        ti = np.minimum(ti, max_ti)

    return ti


def ustar2ti(ustar, ws, max_ti=None):
    """
    Convert friction velocity (u*) to turbulence intensity (TI).

    Parameters
    ----------
    ustar : numpy.ndarray
        Friction velocity.
    ws : numpy.ndarray
        Wind speed.
    max_ti : float, optional
        Upper limit of the computed TI values.

    Returns
    -------
    ti : numpy.ndarray
        Turbulence intensity.

    :group: utils

    """

    ti = (ustar / FC.KAPPA) / ws
    if max_ti is not None:
        ti = np.minimum(ti, max_ti)

    return ti
