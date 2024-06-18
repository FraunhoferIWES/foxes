import numpy as np
from scipy.spatial.distance import cdist


def random_xy_square(
    n,
    min_dist=500,
    xmax_ini=None,
    growth=1.02,
    seed=None,
    verbosity=1,
):
    """
    Creates random xy positions within a square,
    with mean (0, 0)

    Parameters
    ----------
    n: int
        The number of positions
    min_dist: float
        The minimal distance between any two positions
    xmax_ini: float, optional
        The initial maximal distance of any coordinates
    growth: float
        The growth factor of the initial radius, must be
        greater 1
    seed: int, optional
        The random seed
    verbosity: int
        The verbosity level. 0 = silent

    Returns
    -------
    xy: numpy.ndarray
        The positions, shape: (n, 2)

    :group: utils

    """
    if seed:
        np.random.seed(seed)

    xmax = np.sqrt(n) * min_dist if xmax_ini is None else xmax_ini
    xy = np.random.uniform(0, xmax, (n, 2))
    while True:
        dists = cdist(xy, xy)
        np.fill_diagonal(dists, np.inf)
        sel = np.unique(np.where(dists < min_dist)[0])
        if not len(sel):
            break
        if verbosity > 0:
            print(f"Re-generating coordinates: {len(sel)}, xmax = {xmax:.1f}")
        xmax *= growth
        xy[sel] = np.random.uniform(0, xmax, (len(sel), 2))
    return xy - np.mean(xy, axis=0)[None, :]
