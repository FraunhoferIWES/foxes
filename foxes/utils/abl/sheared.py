def calc_ws(
    height: float,
    height0: float,
    WS0: float,
    shear: float,
) -> float:
    """
    Calculate wind speeds at given height

    Parameters
    ----------
    height
        The evaluation height
    height0
        Reference height
    WS0
        Reference wind speed
    shear
        Shear exponent

    Returns
    -------
    ws
        The wind speed


    """
    return WS0 * (height / height0) ** shear
