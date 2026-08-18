from typing import Any

import numpy as np
import pandas as pd

from foxes.utils import Dict
from foxes.core import Turbine, TurbineType, WindFarm
from foxes.models import ModelBook
from foxes.models.farm_controllers import BasicFarmController, OpFlagController
import foxes.variables as FV
import foxes.constants as FC


def read_turbine_types(
    wio_farm: dict[str, Any],
    mbook: ModelBook,
    ws_exp_P: int,
    ws_exp_ct: int,
    verbosity: int,
) -> dict[int, str]:
    """
    Reads the turbine type from windio

    Parameters
    ----------
    wio_farm
        The windio farm data
    mbook
        The model book
    ws_exp_P
        The REWS exponent for power
    ws_exp_ct
        The REWS exponent for ct
    verbosity
        The verbosity level, 0=silent

    Returns
    -------
    ttypes
        Mapping from turbine type key to turbine
        type name in the model book


    """

    def _print(*args: Any, level: int = 1, **kwargs: Any) -> None:
        if verbosity >= level:
            print(*args, **kwargs)

    if "turbine_types" not in wio_farm:
        wio_farm["turbine_types"] = Dict(
            {0: wio_farm["turbines"]}, _name="turbine_types"
        )

    ttypes = {}
    for k, wio_trbns in wio_farm["turbine_types"].items():
        tname = wio_trbns.pop_item("name")
        ttypes[k] = tname
        _print("    Reading turbine type", k, level=3)
        _print("      Name:", tname, level=3)
        _print("      Contents:", [k for k in wio_trbns.keys()], level=3)

        # read performance:
        performance = wio_trbns["performance"]
        _print("        Reading performance", level=3)
        _print("          Contents:", [k for k in performance.keys()], level=3)

        # P, ct data:
        if "power_curve" in performance:
            power_curve = performance["power_curve"]
            _print("            Reading power_curve", level=3)
            _print("              Contents:", [k for k in power_curve.keys()], level=3)
            P = power_curve["power_values"]
            ws_P = power_curve["power_wind_speeds"]
            ct_curve = performance["Ct_curve"]
            data_P = pd.DataFrame(data={"ws": ws_P, "P": P})

            P_max = np.max(data_P["P"])
            for P_unit, P_scale in [("W", 1e6), ("kW", 1e3), ("MW", 1)]:
                if P_max / P_scale >= 1:
                    break
            assert P_max / P_scale >= 1 and P_max / P_scale < 1000, (
                f"Failed to determin P_unit for max power {P_max}"
            )
            _print(f"              Determined P_unit = {P_unit}", level=3)

            _print("            Reading Ct_curve", level=3)
            _print("              Contents:", [k for k in ct_curve.keys()], level=3)
            ct = ct_curve["Ct_values"]
            ws_ct = ct_curve["Ct_wind_speeds"]
            data_ct = pd.DataFrame(data={"ws": ws_ct, "ct": ct})

            def _get_wse_var(wse: int) -> str:
                if wse not in [1, 2, 3]:
                    raise ValueError(
                        f"Expecting wind speed exponent 1, 2 or 3, got {wse}"
                    )
                return FV.REWS if wse == 1 else (FV.REWS2 if wse == 2 else FV.REWS3)

            _print(f"            Creating model '{tname}'", level=3)
            _print("              Turbine type class: PCtFomTwo", level=3)
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
                P_unit=P_unit,
            )
            _print("               ", mbook.turbine_types[tname], level=3)

        # P, ct data:
        elif "Cp_curve" in performance:
            cp_curve = performance["Cp_curve"]
            _print("            Reading Cp_curve", level=3)
            _print("              Contents:", [k for k in cp_curve.keys()], level=3)
            cp = cp_curve["Cp_values"]
            ws_cp = cp_curve["Cp_wind_speeds"]
            ct_curve = performance["Ct_curve"]
            _print("            Reading Ct_curve", level=3)
            _print("              Contents:", [k for k in ct_curve.keys()], level=3)
            ct = ct_curve["Ct_values"]
            ws_ct = ct_curve["Ct_wind_speeds"]

            data_cp = pd.DataFrame(data={"ws": ws_cp, "cp": cp})
            data_ct = pd.DataFrame(data={"ws": ws_ct, "ct": ct})

            _print(f"            Creating model '{tname}'", level=3)
            _print("              Turbine type class: CpCtFromTwo", level=3)
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
            raise KeyError("Expecting either 'power_curve' or 'Cp_curve'")

    return ttypes


def read_layout(
    lname: str,
    ldict: Dict[str, Any],
    farm: WindFarm,
    ttypes: dict[int, str],
    fname: str | None = None,
    verbosity: int = 1,
) -> None:
    """
    Read wind farm layout from windio input

    Parameters
    ----------
    lname
        The layout name
    ldict
        The layout data
    farm
        The wind farm
    ttypes
        Mapping from turbine type key to turbine
        type name in the model book
    fname
        Name of the sub-farm, if any
    verbosity
        The verbosity level, 0=silent


    """
    if verbosity > 2:
        print(f"        Reading '{lname}'")
    cdict = ldict["coordinates"]
    tmap = ldict.get_item("turbine_types", None)
    if verbosity > 2:
        print("          Turbine type map:", tmap)
    for i, xy in enumerate(zip(cdict["x"], cdict["y"])):
        tt = ttypes[tmap[i] if tmap is not None else 0]
        farm.add_turbine(
            Turbine(
                xy=np.array(xy),
                turbine_models=[tt],
                wind_farm_name=fname,
            ),
            verbosity=verbosity - 3,
        )
    if verbosity > 2:
        print(f"          Total number of turbines: {farm.n_turbines}")


def read_farm(wio_dict: dict[str, Any], mbook: ModelBook, verbosity: int) -> WindFarm:
    """
    Reads the wind farm information

    Parameters
    ----------
    wio_dict
        The windio data
    verbosity
        The verbosity level, 0=silent

    Returns
    -------
    farm
        The wind farm


    """
    if not isinstance(wio_dict["wind_farm"], list):
        wio_farms = [wio_dict["wind_farm"]]
    else:
        wio_farms = wio_dict["wind_farm"]

    farm = WindFarm()
    ws_exp_P = None
    operating: list[np.ndarray] | None = None
    for index, wio_farm in enumerate(wio_farms):
        fname = wio_farm.pop_item("name", None)
        if verbosity > 1:
            indx_str = f" [{index}]" if len(wio_farms) > 1 else ""
            print(f"Reading wind farm{indx_str}")
            print("  Name:", fname)
            print("  Contents:", [k for k in wio_farm.keys()])

        # find REWS exponents:
        if ws_exp_P is None:
            try:
                rotor_averaging = wio_dict["attributes"]["analysis"]["rotor_averaging"]
                ws_exp_P = rotor_averaging["wind_speed_exponent_for_power"]
                ws_exp_ct = rotor_averaging["wind_speed_exponent_for_ct"]
            except KeyError:
                ws_exp_P = 1
                ws_exp_ct = 1

        # create farm controller:
        if FV.OPERATING in wio_farm:
            op_dims, optg = wio_farm.pop_item(FV.OPERATING)
            assert (
                len(op_dims) == 2
                and op_dims[1] == FC.TURBINE
                and op_dims[0] in [FC.STATE, FC.TIME]
            ), f"Expecting operating field to have dims (state, turbine), got {op_dims}"
            if operating is None:
                if index > 0:
                    operating = [np.ones((optg.shape[0], farm.n_turbines), dtype=bool)]
                else:
                    operating = []
            operating.append(optg)

        # read turbine type:
        ttypes = read_turbine_types(wio_farm, mbook, ws_exp_P, ws_exp_ct, verbosity)

        # read layouts and create wind farm:
        wfarm = wio_farm["layouts"]
        if isinstance(wfarm, dict):
            if "coordinates" in wfarm:
                wfarm = {"0": wfarm}
            layouts: Dict[str, Any] = Dict(wfarm, _name=wio_farm.name + ".layouts")
        else:
            layouts = Dict(
                {str(i): lf for i, lf in enumerate(wfarm)},
                _name=wio_farm.name + ".layouts",
            )
        if verbosity > 2:
            print("    Reading layouts")
            print("      Contents:", [k for k in layouts.keys()])
        for lname, ldict in layouts.items():
            read_layout(lname, ldict, farm, ttypes, fname=fname, verbosity=verbosity)

    if operating is None:
        mbook.farm_controllers["farm_cntrl"] = BasicFarmController()
    else:
        operating_data = np.concatenate(operating, axis=1)
        mbook.farm_controllers["farm_cntrl"] = OpFlagController(operating_data)
    if verbosity > 1:
        print(
            f"Farm controller type: {type(mbook.farm_controllers['farm_cntrl']).__name__}"
        )

    return farm


def read_n_turbines(wio_dict: dict[str, Any]) -> int:
    """
    Reads the number of turbines from windio input

    Parameters
    ----------
    wio_dict
        The windio data

    Returns
    -------
    n_turbines
        The number of turbines


    """
    wio_farm = wio_dict["wind_farm"]
    wfarm = wio_farm["layouts"]
    layouts: Dict[str, Any]
    if isinstance(wfarm, dict):
        if "coordinates" in wfarm:
            wfarm = {"0": wfarm}
        layouts = Dict(wfarm, _name=wio_farm.name + ".layouts")
    else:
        layouts = Dict(
            {str(i): lf for i, lf in enumerate(wfarm)},
            _name=wio_farm.name + ".layouts",
        )
    n_turbines = 0
    for ldict in layouts.values():
        n_turbines += len(ldict["coordinates"]["x"])

    return n_turbines


def read_hub_heights(wio_dict: dict[str, Any]) -> list[float]:
    """
    Reads the hub heights from windio input

    Parameters
    ----------
    wio_dict
        The windio data

    Returns
    -------
    hub_heights
        The hub heights of all turbines


    """
    wio_farm = wio_dict["wind_farm"]
    if "turbine_types" not in wio_farm:
        wio_farm["turbine_types"] = Dict(
            {0: wio_farm["turbines"]}, _name="turbine_types"
        )

    hub_heights = [tt["hub_height"] for tt in wio_farm["turbine_types"].values()]

    return hub_heights


def read_rotor_diameters(wio_dict: dict[str, Any]) -> list[float]:
    """
    Reads the rotor diameters from windio input

    Parameters
    ----------
    wio_dict
        The windio data

    Returns
    -------
    rotor_diameters
        The rotor diameters of all turbines


    """
    wio_farm = wio_dict["wind_farm"]
    if "turbine_types" not in wio_farm:
        wio_farm["turbine_types"] = Dict(
            {0: wio_farm["turbines"]}, _name="turbine_types"
        )

    rotor_diameters = [
        tt["rotor_diameter"] for tt in wio_farm["turbine_types"].values()
    ]

    return rotor_diameters
