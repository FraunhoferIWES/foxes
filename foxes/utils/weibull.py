import numpy as np

def weibull_weights(ws, ws_deltas, A, k):
    """
    Computes the weibull weights for given wind speeds
    
    Parameters
    ----------
    ws: numpy.ndarray
        The wind speed bin centre values
    ws_deltas: numpy.ndarray
        The wind speed bin widths, same shape as ws
    A: numpy.ndarray
        The Weibull scale parameters, same shape as ws
    k: numpy.ndarray
        The Weibull shape parameters, same shape as ws

    Returns
    -------
    weights: numpy.ndarray
        The weights, same shape as ws
    
    :group: utils

    """
    wsA = ws / A
    return ws_deltas * ( k / A * wsA ** (k - 1) * np.exp(-(wsA**k)) )
