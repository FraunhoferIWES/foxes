def calc_ws(height, height0, WS0, shear):
    """
    Calculate wind speeds at given height

    Parameters
    ----------
    height: float
        The evaluation height
    height0: float
        Reference height
    WS0: float
        Reference wind speed
    shear: float
        Shear exponent

    Returns
    -------
    ws: float
        The wind speed

    :group: utils.abl.sheared

    """
    return WS0 * (height / height0) ** shear
