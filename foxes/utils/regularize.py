from typing import cast

import numpy as np


def sqrt_reg(x: np.ndarray, x0: float = 0.01) -> np.ndarray:
    """
    A regularized sqrt function, producing
    non-zero values also for smallish negative x.

    Parameters
    ----------
    x
        The x values to evaluate
    x0
        Parameter where to start the smoothing

    Returns
    -------
    out
        The regularized sqrt(x) results

    :group: utils

    """
    b = x0 * (1 - np.log(x0))
    y = np.exp((x - b) / x0)
    return cast(np.ndarray, np.sqrt(np.where(x < x0, y, x)))
