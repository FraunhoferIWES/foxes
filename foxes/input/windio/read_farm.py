import numpy as np
import pandas as pd

from foxes.utils import Dict
from foxes.core import Turbine, TurbineType, WindFarm

def read_turbine_type(turbines, data, verbosity):
    tname = turbines.pop("name")
    if verbosity > 1:
        print("    Reading turbines")
        print("      Name:", tname)
        print("      Contents:", [k  for k in turbines.keys()])
    
    # read performance:
    performance = Dict(turbines["performance"], name="performance")
    if verbosity > 1:
        print("        Reading performance")
        print("          Contents:", [k  for k in performance.keys()])
    
    # P, ct data:
    if "power_curve" in performance:
        power_curve = Dict(performance["power_curve"], name="power_curve")
        if verbosity > 1:
            print("            Reading power_curve") 
            print("              Contents:", [k  for k in power_curve.keys()])
        P = power_curve["power_values"]
        ws_P = power_curve["power_wind_speeds"]
        ct_curve = Dict(performance["Ct_curve"], name="Ct_values")
        if verbosity > 1:
            print("            Reading Ct_curve") 
            print("              Contents:", [k  for k in ct_curve.keys()])
        ct = ct_curve["Ct_values"]
        ws_ct = ct_curve["Ct_wind_speeds"]
        
        data_P = pd.DataFrame(data={"ws": ws_P, "P": P})
        data_ct = pd.DataFrame(data={"ws": ws_ct, "ct": ct})

        if verbosity > 1:
            print(f"            Creating model '{tname}'")
            print(f"              Turbine type class: PCtFromTwo")
        data["mbook"].turbine_types[tname] = TurbineType.new(
            ttype_type="PCtFromTwo", 
            data_source_P=data_P,
            data_source_ct=data_ct,
            col_ws_P_file="ws",
            col_ws_ct_file="ws",
            col_P="P",
            col_ct="ct",
            H=turbines["hub_height"],
            D=turbines["rotor_diameter"],
        )

    # P, ct data:
    elif "Cp_curve" in performance:
        cp_curve = Dict(performance["Cp_curve"], name="Cp_curve")
        if verbosity > 1:
            print("            Reading Cp_curve") 
            print("              Contents:", [k  for k in cp_curve.keys()])
        cp = cp_curve["Cp_values"]
        ws_cp = cp_curve["Cp_wind_speeds"]
        ct_curve = Dict(performance["Ct_curve"], name="Ct_values")
        if verbosity > 1:
            print("            Reading Ct_curve") 
            print("              Contents:", [k  for k in ct_curve.keys()])
        ct = ct_curve["Ct_values"]
        ws_ct = ct_curve["Ct_wind_speeds"]
        
        data_cp = pd.DataFrame(data={"ws": ws_cp, "cp": cp})
        data_ct = pd.DataFrame(data={"ws": ws_ct, "ct": ct})

        if verbosity > 1:
            print(f"            Creating model '{tname}'")
            print(f"              Turbine type class: CpCtFromTwo")
        data["mbook"].turbine_types[tname] = TurbineType.new(
            ttype_type="CpCtFromTwo", 
            data_source_cp=data_cp,
            data_source_ct=data_ct,
            col_ws_cp_file="ws",
            col_ws_ct_file="ws",
            col_cp="cp",
            col_ct="ct",
            H=turbines["hub_height"],
            D=turbines["rotor_diameter"],
        )
    
    else:
        raise KeyError(f"Expecting either 'power_curve' or 'Cp_curve'")

    return tname

def read_layout(lname, ldict, data, ttype, verbosity=1):
    """
    Read wind farm layout from windio input

    Parameters
    ----------
    lname: str
        The layout name
    ldict: dict
        The layout data
    fields: dict
        The fields data
    dims: dict
        The dimensions data
    ttype: str
        Name of the turbine type model
    verbosity: int
        The verbosity level
    
    Returns
    -------
    states: foxes.core.States
        The states object
    
    :group: input.windio

    """
    if verbosity > 1:
        print(f"        Reading '{lname}'")
    cdict = Dict(ldict["coordinates"], name="coordinates")
    farm = WindFarm()
    for xy in zip(cdict["x"], cdict["y"]):
        farm.add_turbine(
            Turbine(xy=np.array(xy), turbine_models=[ttype]),
            verbosity=0,
        )
    if verbosity > 1:
        print(f"          Added {farm.n_turbines} turbines of type '{ttype}'")
    data["farm"] = farm
        