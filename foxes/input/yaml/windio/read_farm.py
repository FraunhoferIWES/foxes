import numpy as np
import pandas as pd

from foxes.utils import Dict
from foxes.core import Turbine, TurbineType
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
        tname = wio_trbns.pop("name")
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
