import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from foxes.core import WindFarm
from foxes.models import ModelBook
from foxes.input.states import StatesTable
from foxes.input.farm_layout import add_from_df
from foxes.models.turbine_types import PCt_from_two
import foxes.constants as FC
import foxes.variables as FV

def read_resource(res_yaml, fixed_vars=None, **kwargs):
    """
    Reads a WindIO energy resource

    Parameters
    ----------
    res_yaml : str
        Path to the yaml file
    fixed_vars : dict, optional
        Additional fixes variables that do 
        not occur in the yaml
    kwargs : dict, optional
        Additional arguments for StatesTable

    Returns
    -------
    states: foxes.states.StatesTable
        The uniform states

    """
    res_yaml = Path(res_yaml)

    with open(res_yaml, 'r') as file:
        res = yaml.load(file, Loader=yaml.loader.BaseLoader)
    
    N = 0
    for v in ["wind_direction", "wind_speed", "turbulence_intensity"]:
        if v in res["wind_resource"]:
            N = max(N, len(res["wind_resource"][v]))
    data = pd.DataFrame(index=range(N))
    data.index.name = "state"

    vmap = {
        "wind_direction": FV.WD,
        "wind_speed": FV.WS,
        "turbulence_intensity": FV.TI,
        "air_density": FV.RHO,
        "probability": FV.WEIGHT
    }

    for v, d in res["wind_resource"].items():

        if v == "probability":
            d = d["data"]
        elif v == "turbulence_intensity":
            if len(d["dims"]) == 0:
                d = d["data"]
            elif d["dims"] == ["%"]:
                d = d["data"] / 100.0
            else:
                raise ValueError(f"Unknown dimensions for variable '{v}'. Know '[]' or '[%]', found: {d['dims']}")
        
        data[vmap[v]] = np.array(d, dtype=FC.DTYPE) if len(d) > 1 else d[0]
    
    ovars = [v for v in data.columns if v != FV.WEIGHT]
    if fixed_vars is not None:
        ovars.append([v for v in fixed_vars.keys() if v not in data.columns])

    return StatesTable(
        data,
        output_vars=ovars,
        fixed_vars=fixed_vars,
        **kwargs
    )

def read_site(site_yaml, **kwargs):
    """
    Reads a WindIO site

    Parameters
    ----------
    site_yaml : str
        Path to the yaml file
    kwargs : dict, optional
        Additional arguments for read_resource

    Returns
    -------
    states: foxes.states.StatesTable
        The uniform states

    """
    site_yaml = Path(site_yaml)

    with open(site_yaml, 'r') as file:
        site = yaml.load(file, Loader=yaml.loader.BaseLoader)

    res_yaml = site["energy_resource"]
    if res_yaml[0] == ".":
        res_yaml = (site_yaml.parent/res_yaml).resolve()
    states = read_resource(res_yaml, **kwargs)

    return states

def read_farm(farm_yaml, mbook=None, layout=-1):
    """
    Reads a WindIO wind farm

    Parameters
    ----------
    farm_yaml : str
        Path to the yaml file
    mbook: foxes.ModelBook, optional
        The model book to start from
    layout : str or int
        The layout choice

    Returns
    -------
    farm : foxes.WindFarm
        The wind farm

    """
    mbook = ModelBook() if mbook is None else mbook
    farm_yaml = Path(farm_yaml)
    print(farm_yaml)

    with open(farm_yaml, 'r') as file:
        fdict = yaml.load(file, Loader=yaml.loader.BaseLoader)
    
    print("READ FARM\n", fdict)

    if isinstance(layout, str):
        layout = fdict['layouts'][layout]
    else:
        lname = list(fdict['layouts'].keys())[layout]
        layout = fdict['layouts'][lname]
    print(layout)

    import json
    print(json.dumps(fdict, sort_keys=True, indent=4))
    quit()

    x = np.array(layout["coordinates"]["x"], dtype=FC.DTYPE)
    y = np.array(layout["coordinates"]["y"], dtype=FC.DTYPE)
    N = len(x)
    ldata = pd.DataFrame(index=range(N))
    ldata.index.name = "index"
    ldata["x"] = x
    ldata["y"] = y

    print(ldata)

    ct_ws = np.array(fdict["Ct_curve"]["Ct_wind_speeds"], dtype=FC.DTYPE)
    ct_data = pd.DataFrame(index=ct_ws)
    ct_data.index.name = "ws"
    ct_data["ct"] = np.array(fdict["Ct_curve"]["Ct_values"], dtype=FC.DTYPE)

    ct_ws = np.array(fdict["Ct_curve"]["Ct_wind_speeds"], dtype=FC.DTYPE)
    ct_data = pd.DataFrame(index=ct_ws)
    ct_data.index.name = "ws"
    ct_data["ct"] = np.array(fdict["Ct_curve"]["Ct_values"], dtype=FC.DTYPE)

    #mbook.turbine_types["windio_turbine"] = 

    farm = WindFarm(name=fdict["name"])
    add_from_df(farm, ldata, col_x="x", col_y="y")

    quit()
        
def read_case(case_yaml, site_pars={}, farm_pars={}):
    """
    Reads a WindIO case

    Parameters
    ----------
    case_yaml : str
        Path to the yaml file
    site_pars : dict
        Additional arguments for read_site
    farm_pars : dict
        Additional arguments for read_farm

    Returns
    -------


    """
    case_yaml = Path(case_yaml)

    with open(case_yaml, 'r') as file:
        case = yaml.load(file, Loader=yaml.loader.BaseLoader)
    
    site_yaml = case["site"]
    if site_yaml[0] == ".":
        site_yaml = (case_yaml.parent/site_yaml).resolve()
    states = read_site(site_yaml, **site_pars)

    farm_yaml = case["wind_farm"]
    if farm_yaml[0] == ".":
        farm_yaml = (case_yaml.parent/farm_yaml).resolve()
    farm = read_farm(farm_yaml, **farm_pars)

    print(case)
    quit()

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("case_yaml", help="The case yaml file")
    args = parser.parse_args()

    read_case(args.case_yaml)