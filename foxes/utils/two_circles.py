import numpy as np


def area(r1, r2, d):
    """
    Calculates the intersection of two
    circles with radii r1, r2 and a centre
    point distance d.

    Make sure that

        1) r1 >= r2
        2) r1 - r2 <= d <= r1 + r2

    These conditions are assumed but not checked
    within the function.

    Source:
    https://diego.assencio.com/?index=8d6ca3d82151bad815f78addf9b5c1c6

    Parameters
    ----------
    r1: float or numpy.ndarray
        The radius of circle 1
    r2: float or numpy.ndarray
        The radius of circle 2
    d: float or numpy.ndarray
        The distance between the centre points
        of the two circles

    Returns
    -------
    area: float or numpy.ndarray
        The intersectional area

    :group: utils.two_circles

    """
    d1 = (r1**2 - r2**2 + d**2) / (2 * d)
    d2 = d - d1

    a = np.maximum(np.minimum(d1 / r1, 1.0), -1)
    b = np.maximum(r1**2 - d1**2, 0.0)
    A1 = r1**2 * np.arccos(a) - d1 * np.sqrt(b)

    a = np.maximum(np.minimum(d2 / r2, 1.0), -1)
    b = np.maximum(r2**2 - d2**2, 0.0)
    A2 = r2**2 * np.arccos(a) - d2 * np.sqrt(b)

    """
    A1 = r1**2 * np.arccos(d1/r1) - d1 * np.sqrt(r1**2 - d1**2)
    A2 = r2**2 * np.arccos(d2/r2) - d2 * np.sqrt(r2**2 - d2**2)
    """

    return A1 + A2


def calc_area(r1, r2, d):
    """
    Calculates the intersection of two circles.

    All parameters should have the same shape,
    or be broadcastable to one another.

    Parameters
    ----------
    r1: numpy.ndarray
        The radius of circle 1
    r2: numpy.ndarray
        The radius of circle 2
    d: numpy.ndarray
        The distance between the centre points
        of the two circles

    Returns
    -------
    area: numpy.ndarray
        The intersectional area

    :group: utils.two_circles

    """

    # prepare:
    out = np.zeros_like(d)

    # condition d < r1 + r2:
    sel0 = d < r1 + r2
    if np.any(sel0):
        # condition r1 >= r2:
        sela = r1 >= r2
        selr = sel0 & sela
        if np.any(selr):
            # condition d <= r1 - r2:
            selb = d <= r1 - r2
            seld = selr & selb
            if np.any(seld):
                out[seld] = np.pi * r2[seld] ** 2

            # condition d > r1 - r2:
            seld = selr & (~selb)
            if np.any(seld):
                out[seld] = area(r1[seld], r2[seld], d[seld])

        # condition r1 < r2:
        selr = sel0 & (~sela)
        if np.any(selr):
            # condition d <= r2 - r1:
            selb = d <= r2 - r1
            seld = selr & selb
            if np.any(seld):
                out[seld] = np.pi * r1[seld] ** 2

            # condition d > r2 - r1:
            seld = selr & (~selb)
            if np.any(seld):
                out[seld] = area(r2[seld], r1[seld], d[seld])

    return out
