import numpy as np
import pandas as pd

from foxes.utils import Dict
from foxes.core import Turbine, TurbineType
import foxes.variables as FV


def read_turbine_types(wio_farm, algo_dict, ws_exp_P, ws_exp_ct, verbosity):
    """
    Reads the turbine type from windio

    Parameters
    ----------
    wio_farm: dict
        The windio farm data
    algo_dict: dict
        The algorithm dictionary
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

    :group: input.windio

    """
    if "turbine_types" not in wio_farm:
        wio_farm["turbine_types"] = {0: wio_farm["turbines"]}

    ttypes = {}
    for k, wio_trbns in wio_farm["turbine_types"].items():
        tname = wio_trbns.pop("name")
        ttypes[k] = tname
        if verbosity > 2:
            print("    Reading turbine type", k)
            print("      Name:", tname)
            print("      Contents:", [k for k in wio_trbns.keys()])

        # read performance:
        performance = Dict(wio_trbns["performance"], name="performance")
        if verbosity > 2:
            print("        Reading performance")
            print("          Contents:", [k for k in performance.keys()])

        # P, ct data:
        if "power_curve" in performance:
            power_curve = Dict(performance["power_curve"], name="power_curve")
            if verbosity > 2:
                print("            Reading power_curve")
                print("              Contents:", [k for k in power_curve.keys()])
            P = power_curve["power_values"]
            ws_P = power_curve["power_wind_speeds"]
            ct_curve = Dict(performance["Ct_curve"], name="Ct_values")
            if verbosity > 2:
                print("            Reading Ct_curve")
                print("              Contents:", [k for k in ct_curve.keys()])
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

            if verbosity > 2:
                print(f"            Creating model '{tname}'")
                print(f"              Turbine type class: PCtFromTwo")
            algo_dict["mbook"].turbine_types[tname] = TurbineType.new(
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
            if verbosity > 2:
                print("               ", algo_dict["mbook"].turbine_types[tname])

        # P, ct data:
        elif "Cp_curve" in performance:
            cp_curve = Dict(performance["Cp_curve"], name="Cp_curve")
            if verbosity > 2:
                print("            Reading Cp_curve")
                print("              Contents:", [k for k in cp_curve.keys()])
            cp = cp_curve["Cp_values"]
            ws_cp = cp_curve["Cp_wind_speeds"]
            ct_curve = Dict(performance["Ct_curve"], name="Ct_values")
            if verbosity > 2:
                print("            Reading Ct_curve")
                print("              Contents:", [k for k in ct_curve.keys()])
            ct = ct_curve["Ct_values"]
            ws_ct = ct_curve["Ct_wind_speeds"]

            data_cp = pd.DataFrame(data={"ws": ws_cp, "cp": cp})
            data_ct = pd.DataFrame(data={"ws": ws_ct, "ct": ct})

            if verbosity > 2:
                print(f"            Creating model '{tname}'")
                print(f"              Turbine type class: CpCtFromTwo")
            algo_dict["mbook"].turbine_types[tname] = TurbineType.new(
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

        else:
            raise KeyError(f"Expecting either 'power_curve' or 'Cp_curve'")

    return ttypes


def read_layout(lname, ldict, algo_dict, ttypes, verbosity=1):
    """
    Read wind farm layout from windio input

    Parameters
    ----------
    lname: str
        The layout name
    ldict: dict
        The layout data
    algo_dict: dict
        The algorithm dictionary
    ttypes: dict
        Mapping from turbine type key to turbine
        type name in the model book
    verbosity: int
        The verbosity level, 0=silent

    Returns
    -------
    states: foxes.core.States
        The states object

    :group: input.windio

    """
    if verbosity > 2:
        print(f"        Reading '{lname}'")
    cdict = Dict(ldict["coordinates"], name="coordinates")
    farm = algo_dict["farm"]
    tmap = ldict.get("turbine_types", None)
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
