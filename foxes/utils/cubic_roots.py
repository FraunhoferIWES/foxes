import numpy as np


def cubic_roots(a0, a1, a2, a3=None):
    """
    Calculate real roots of polynomials of degree 3.

    Convention:
    f(x) = a[3]*x**3 + a[2]*x**2 + a[1]*x + a[0]

    In contrast to numpy's "root" function
    this works fast for an array of polynomials,
    so you spare yourself looping over them.

    Source: https://github.com/opencv/opencv/blob/master/modules/calib3d/src/polynom_solver.cpp

    Parameters
    ----------
    a0: numpy.ndarray
        The coefficients a[0]
    a1: numpy.ndarray
        The coefficients a[1]
    a2: numpy.ndarray
        The coefficients a[2]
    a3: numpy.ndarray
        The coefficients a[3], or None for ones

    Returns
    -------
    roots: numpy.ndarray
        The real roots of the polynomial,
        shape: (n_a0, 3). If one root only
        the two last columns will be np.nan

    :group: utils

    """

    N = len(a0)
    out = np.full([N, 3], np.nan)

    # Calculate the normalized form x^3 + a2 * x^2 + a1 * x + a0 = 0
    b_a = a2 if a3 is None else a2 / a3
    b_a2 = b_a * b_a
    c_a = a1 if a3 is None else a1 / a3
    d_a = a0 if a3 is None else a0 / a3

    # Solve the cubic equation
    Q = (3 * c_a - b_a2) / 9
    R = (9 * b_a * c_a - 27 * d_a - 2 * b_a * b_a2) / 54
    Q3 = Q * Q * Q
    D = Q3 + R * R
    b_a_3 = (1.0 / 3.0) * b_a

    sel = Q == 0.0
    if np.any(sel):
        o = out[sel, 0]

        sel2 = R == 0.0
        if np.any(sel2):
            o[sel2] = -b_a_3[sel][sel2]

        if np.any(~sel2):
            o[~sel2] = np.pow(2 * R[sel][~sel2], 1 / 3.0) - b_a_3[sel][~sel2]

        out[sel, 0] = o

    sel = D <= 0.0
    if np.any(sel):
        # Three real roots
        theta = np.arccos(R[sel] / np.sqrt(-Q3[sel]))
        sqrt_Q = np.sqrt(-Q[sel])

        out[sel, 0] = 2 * sqrt_Q * np.cos(theta / 3.0) - b_a_3[sel]
        out[sel, 1] = 2 * sqrt_Q * np.cos((theta + 2 * np.pi) / 3.0) - b_a_3[sel]
        out[sel, 2] = 2 * sqrt_Q * np.cos((theta + 4 * np.pi) / 3.0) - b_a_3[sel]

    return out


def test_cubic_roots(roots, a0, a1, a2, a3=None, tol=1.0e-12):
    """
    Test the cubic roots results

    Parameters
    ----------
    roots: numpy.ndarray
        The roots to test, shape: (n_a0, 3)
    a0: numpy.ndarray
        The coefficients a[0]
    a1: numpy.ndarray
        The coefficients a[1]
    a2: numpy.ndarray
        The coefficients a[2]
    a3: numpy.ndarray
        The coefficients a[3], or None for ones

    """

    N = len(a0)
    for n in range(N):
        c0 = a0[n]
        c1 = a1[n]
        c2 = a2[n]
        c3 = a3[n]

        print(f"Polynomial {n}: a = {(c0,c1,c2,c3)}")

        rts = np.unique(roots[n])
        rts = rts[~np.isnan(rts)]

        for x in rts:
            f = c0 + c1 * x + c2 * x**2 + c3 * x**3
            ok = np.abs(f) <= tol

            print(f"  root x = {x}: f(x) = {f}     {'OK' if ok else 'FAILED'}")

            if not ok:
                raise Exception("NOT OK!")

        if len(rts) == 0:
            print("  no real roots.")


if __name__ == "__main__":
    N = 100
    a0 = np.random.uniform(-10.0, 10.0, N)
    a1 = np.random.uniform(-10.0, 10.0, N)
    a2 = np.random.uniform(-10.0, 10.0, N)
    a3 = np.random.uniform(1.0, 10.0, N)

    roots = cubic_roots(a0, a1, a2, a3)

    test_cubic_roots(roots, a0, a1, a2, a3)
