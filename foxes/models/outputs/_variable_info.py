"""Variable labels shared by stateless scientific renderers."""

import foxes.variables as FV


def get_data_info(dname: str, unit: str | None = None) -> tuple[str, str]:
    """Return a scientific variable's default title and axis label.

    Parameters
    ----------
    dname
        FOXES variable name, including ambient variants.
    unit
        Dataset unit overriding the default label. Power has no default unit;
        supply the unit resolved from the turbine types.

    Returns
    -------
    title
        Human-readable quantity name; unknown names are returned unchanged.
    legend
        Axis or legend label with established units. Ambient labels retain
        the ``AMB_`` prefix; unknown names are returned unchanged.
    """
    if unit is not None:
        title, _ = get_data_info(dname)
        return title, f"{dname} [{unit}]"

    if dname == FV.D:
        return "Rotor diameter", f"{FV.D} [m]"
    if dname == FV.H:
        return "Hub height", f"{FV.H} [m]"
    if dname == FV.WS:
        return "Wind speed", f"{FV.WS} [m/s]"
    if dname == FV.REWS:
        return "Rotor equivalent wind speed", f"{FV.REWS} [m/s]"
    if dname == FV.REWS2:
        return "Rotor equivalent wind speed (2nd moment)", f"{FV.REWS2} [m/s]"
    if dname == FV.REWS3:
        return "Rotor equivalent wind speed (3rd moment)", f"{FV.REWS3} [m/s]"
    if dname == FV.WD:
        return "Wind direction", f"{FV.WD} [deg]"
    if dname == FV.TI:
        return "Turbulence intensity", f"{FV.TI} [1]"
    if dname == FV.RHO:
        return "Air density", f"{FV.RHO} [kg/m3]"
    if dname == FV.CT:
        return "Thrust coefficient", f"{FV.CT} [1]"
    if dname == FV.P:
        return "Power", FV.P
    if dname == FV.YAW:
        return "Yaw angle", f"{FV.YAW} [deg]"
    if dname == FV.YAWM:
        return "Yaw misalignment", f"{FV.YAWM} [deg]"

    if dname in FV.amb2var:
        title, legend = get_data_info(FV.amb2var[dname])
        return f"Ambient {title.lower()}", f"AMB_{legend}"

    return dname, dname
