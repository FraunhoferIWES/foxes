import numpy as np


def weibull_weights(
    ws: np.ndarray,
    ws_deltas: np.ndarray,
    A: np.ndarray,
    k: np.ndarray,
) -> np.ndarray:
    """
    Computes the weibull weights for given wind speeds

    Parameters
    ----------
    ws
        The wind speed bin centre values
    ws_deltas
        The wind speed bin widths, same shape as ws
    A
        The Weibull scale parameters, same shape as ws
    k
        The Weibull shape parameters, same shape as ws

    Returns
    -------
    weights
        The weights, same shape as ws

    :group: utils

    """
    wsA = ws / A
    return ws_deltas * (k / A * wsA ** (k - 1) * np.exp(-(wsA**k)))
