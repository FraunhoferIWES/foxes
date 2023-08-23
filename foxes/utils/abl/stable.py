import numpy as np
from .neutral import logz as lgz


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

    :group: utils.abl.stable

    """
    return lgz(height, z0)


def psi(height, mol):
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


def ustar(ws_ref, h_ref, z0, mol, kappa=0.41):
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
        The van-Karman constant

    Returns
    -------
    ustar: float
        The friction velocity

    :group: utils.abl.stable

    """
    return ws_ref * kappa / (logz(h_ref, z0) - psi(h_ref, mol))


def calc_ws(height, z0, ustar, psi, kappa=0.41):
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
        The van-Karman constant

    Returns
    -------
    ws: float
        The wind speed

    :group: utils.abl.stable

    """
    return ustar / kappa * (logz(height, z0) - psi)
