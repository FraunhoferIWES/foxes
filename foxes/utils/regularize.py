import numpy as np


def sqrt_reg(x, x0=0.01):
    """
    A regularized sqrt function, producing
    non-zero values also for smallish negative x.

    Parameters
    ----------
    x: numpy.ndarray
        The x values to evaluate
    x0: float
        Parameter where to start the smoothing

    Returns
    -------
    out: numpy.ndarray
        The regularized sqrt(x) results

    :group: utils

    """
    b = x0 * (1 - np.log(x0))
    y = np.exp((x - b) / x0)
    return np.sqrt(np.where(x < x0, y, x))
