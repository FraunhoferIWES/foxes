import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from foxes.input.states import StatesTable
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

def read_farm(farm_yaml):
    """
    Reads a WindIO wind farm

    Parameters
    ----------
    farm_yaml : str
        Path to the yaml file

    Returns
    -------
    farm : foxes.WindFarm
        The wind farm

    """
    farm_yaml = Path(farm_yaml)

    with open(farm_yaml, 'r') as file:
        fdict = yaml.load(file, Loader=yaml.loader.BaseLoader)
    
    print("READ FARM\n", fdict)
    quit()
        
def read_case(case_yaml, **kwargs):
    """
    Reads a WindIO case

    Parameters
    ----------
    case_yaml : str
        Path to the yaml file
    kwargs : dict, optional
        Additional arguments for read_site

    Returns
    -------


    """
    case_yaml = Path(case_yaml)

    with open(case_yaml, 'r') as file:
        case = yaml.load(file, Loader=yaml.loader.BaseLoader)
    
    site_yaml = case["site"]
    if site_yaml[0] == ".":
        site_yaml = (case_yaml.parent/site_yaml).resolve()
    states = read_site(site_yaml, **kwargs)

    farm_yaml = case["wind_farm"]
    if farm_yaml[0] == ".":
        farm_yaml = (case_yaml.parent/farm_yaml).resolve()
    farm = read_farm(farm_yaml)

    print(case)
    quit()

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("case_yaml", help="The case yaml file")
    args = parser.parse_args()

    read_case(args.case_yaml)