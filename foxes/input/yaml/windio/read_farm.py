import numpy as np
import pandas as pd

from foxes.utils import Dict
from foxes.core import Turbine, TurbineType, WindFarm
import foxes.variables as FV


def read_turbine_types(wio_farm, mbook, ws_exp_P, ws_exp_ct, verbosity):
    """
    Reads the turbine type from windio

    Parameters
    ----------
    wio_farm: dict
        The windio farm data
    mbook: foxes.models.ModelBook
        The model book
    ws_exp_P: int
        The REWS exponent for power
    ws_exp_ct: int
        The REWS exponent for ct
    verbosity: int
        The verbosity level, 0=silent

    Returns
    -------
    ttypes: dict
        Mapping from turbine type key to turbine
        type name in the model book

    :group: input.yaml.windio

    """

    def _print(*args, level=1, **kwargs):
        if verbosity >= level:
            print(*args, **kwargs)

    if "turbine_types" not in wio_farm:
        wio_farm["turbine_types"] = {0: wio_farm["turbines"]}

    ttypes = {}
    for k, wio_trbns in wio_farm["turbine_types"].items():
        tname = wio_trbns.pop_item("name")
        ttypes[k] = tname
        _print("    Reading turbine type", k, level=3)
        _print("      Name:", tname, level=3)
        _print("      Contents:", [k for k in wio_trbns.keys()], level=3)

        # read performance:
        performance = Dict(wio_trbns["performance"], name="performance")
        _print("        Reading performance", level=3)
        _print("          Contents:", [k for k in performance.keys()], level=3)

        # P, ct data:
        if "power_curve" in performance:
            power_curve = Dict(performance["power_curve"], name="power_curve")
            _print("            Reading power_curve", level=3)
            _print("              Contents:", [k for k in power_curve.keys()], level=3)
            P = power_curve["power_values"]
            ws_P = power_curve["power_wind_speeds"]
            ct_curve = Dict(performance["Ct_curve"], name="Ct_values")
            _print("            Reading Ct_curve", level=3)
            _print("              Contents:", [k for k in ct_curve.keys()], level=3)
            ct = ct_curve["Ct_values"]
            ws_ct = ct_curve["Ct_wind_speeds"]

            data_P = pd.DataFrame(data={"ws": ws_P, "P": P})
            data_ct = pd.DataFrame(data={"ws": ws_ct, "ct": ct})

            def _get_wse_var(wse):
                if wse not in [1, 2, 3]:
                    raise ValueError(
                        f"Expecting wind speed exponent 1, 2 or 3, got {wse}"
                    )
                return FV.REWS if wse == 1 else (FV.REWS2 if wse == 2 else FV.REWS3)

            _print(f"            Creating model '{tname}'", level=3)
            _print(f"              Turbine type class: PCtFomTwo", level=3)
            mbook.turbine_types[tname] = TurbineType.new(
                ttype_type="PCtFromTwo",
                data_source_P=data_P,
                data_source_ct=data_ct,
                col_ws_P_file="ws",
                col_ws_ct_file="ws",
                col_P="P",
                col_ct="ct",
                H=wio_trbns["hub_height"],
                D=wio_trbns["rotor_diameter"],
                var_ws_ct=_get_wse_var(ws_exp_ct),
                var_ws_P=_get_wse_var(ws_exp_P),
                rho=1.225,
            )
            _print("               ", mbook.turbine_types[tname], level=3)

        # P, ct data:
        elif "Cp_curve" in performance:
            cp_curve = Dict(performance["Cp_curve"], name="Cp_curve")
            _print("            Reading Cp_curve", level=3)
            _print("              Contents:", [k for k in cp_curve.keys()], level=3)
            cp = cp_curve["Cp_values"]
            ws_cp = cp_curve["Cp_wind_speeds"]
            ct_curve = Dict(performance["Ct_curve"], name="Ct_values")
            _print("            Reading Ct_curve", level=3)
            _print("              Contents:", [k for k in ct_curve.keys()], level=3)
            ct = ct_curve["Ct_values"]
            ws_ct = ct_curve["Ct_wind_speeds"]

            data_cp = pd.DataFrame(data={"ws": ws_cp, "cp": cp})
            data_ct = pd.DataFrame(data={"ws": ws_ct, "ct": ct})

            _print(f"            Creating model '{tname}'", level=3)
            _print(f"              Turbine type class: CpCtFromTwo", level=3)
            mbook.turbine_types[tname] = TurbineType.new(
                ttype_type="CpCtFromTwo",
                data_source_cp=data_cp,
                data_source_ct=data_ct,
                col_ws_cp_file="ws",
                col_ws_ct_file="ws",
                col_cp="cp",
                col_ct="ct",
                H=wio_trbns["hub_height"],
                D=wio_trbns["rotor_diameter"],
            )
            _print("               ", mbook.turbine_types[tname], level=3)

        else:
            raise KeyError(f"Expecting either 'power_curve' or 'Cp_curve'")

    return ttypes


def read_layout(lname, ldict, farm, ttypes, verbosity=1):
    """
    Read wind farm layout from windio input

    Parameters
    ----------
    lname: str
        The layout name
    ldict: dict
        The layout data
    farm: foxes.core.WindFarm
        The wind farm
    ttypes: dict
        Mapping from turbine type key to turbine
        type name in the model book
    verbosity: int
        The verbosity level, 0=silent

    :group: input.yaml.windio

    """
    if verbosity > 2:
        print(f"        Reading '{lname}'")
    cdict = Dict(ldict["coordinates"], name="coordinates")
    tmap = ldict.get_item("turbine_types", None)
    if verbosity > 2:
        print(f"          Turbine type map:", tmap)
    for i, xy in enumerate(zip(cdict["x"], cdict["y"])):
        tt = ttypes[tmap[i] if tmap is not None else 0]
        farm.add_turbine(
            Turbine(xy=np.array(xy), turbine_models=[tt]),
            verbosity=verbosity - 3,
        )
    if verbosity > 2:
        print(f"          Added {farm.n_turbines} turbines")


def read_farm(wio_dict, mbook, verbosity):
    """
    Reads the wind farm information

    Parameters
    ----------
    wio_dict: foxes.utils.Dict
        The windio data
    verbosity: int
        The verbosity level, 0=silent

    Returns
    -------
    farm: foxes.core.WindFarm
        The wind farm

    :group: input.yaml.windio

    """
    wio_farm = Dict(wio_dict["wind_farm"], name=wio_dict.name + ".wind_farm")
    if verbosity > 1:
        print("Reading wind farm")
        print("  Name:", wio_farm.pop_item("name", None))
        print("  Contents:", [k for k in wio_farm.keys()])

    # find REWS exponents:
    try:
        rotor_averaging = wio_dict["attributes"]["analysis"]["rotor_averaging"]
        ws_exp_P = rotor_averaging["wind_speed_exponent_for_power"]
        ws_exp_ct = rotor_averaging["wind_speed_exponent_for_ct"]
    except KeyError:
        ws_exp_P = 1
        ws_exp_ct = 1

    # read turbine type:
    ttypes = read_turbine_types(wio_farm, mbook, ws_exp_P, ws_exp_ct, verbosity)

    # read layouts and create wind farm:
    farm = WindFarm()
    wfarm = wio_farm["layouts"]
    if isinstance(wfarm, dict):
        layouts = Dict(wfarm, name=wio_farm.name + ".layouts")
    else:
        layouts = {str(i): l for i, l in enumerate(wfarm)}
        layouts = Dict(layouts, name=wio_farm.name + ".layouts")
    if verbosity > 2:
        print("    Reading layouts")
        print("      Contents:", [k for k in layouts.keys()])
    for lname, ldict in layouts.items():
        read_layout(lname, ldict, farm, ttypes, verbosity)

    return farm
