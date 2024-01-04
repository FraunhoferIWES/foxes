import numpy as np
import pandas as pd

from foxes.core import WindFarm, Algorithm
from foxes.models import ModelBook
from foxes.input.farm_layout import add_from_df
from foxes.models.turbine_types import PCtFromTwo, CpCtFromTwo
from foxes.utils import import_module
import foxes.constants as FC

from .read_states import read_Timeseries, read_StatesTable
from .output import WIOOutput

def read_resource(res, **site_pars):
    """
    Reads a WindIO energy resource

    Parameters
    ----------
    res: dict
        Data from the yaml file
    fixed_vars: dict
        Additional fixes variables that do
        not occur in the yaml
    site_pars: dict, optional
        Additional arguments for read_resource

    Returns
    -------
    states: foxes.states.States
        The states object

    """
    wres = res["wind_resource"]
    if len(wres) != 1:
        raise KeyError(f"Expecting exactly one entry in wind_resource, found: {list(wres.keys())}")

    if "timeseries" in wres:
        return read_Timeseries(wres["timeseries"], **site_pars)
    else:
        return read_StatesTable(wres, **site_pars)

def read_site(site, **site_pars):
    """
    Reads a WindIO site

    Parameters
    ----------
    site_data: dict
        Data from the yaml file
    site_pars: dict, optional
        Additional arguments for read_resource

    Returns
    -------
    states: foxes.states.States
        The states object

    """
    res = site["energy_resource"]
    states = read_resource(res, **site_pars)

    return states

def read_farm(fdict, mbook, tmdict, layout=-1, **kwargs):
    """
    Reads a WindIO wind farm

    Parameters
    ----------
    farm_data: dict
        Data from the yaml file
    mbook: foxes.ModelBook
        The model book
    tmdict: dict
        The turbine model dict
    layout: str or int
        The layout choice
    kwargs: dict, optional
        Additional parameters for add_from_df()

    Returns
    -------
    farm: foxes.WindFarm
        The wind farm

    """
    if isinstance(layout, str):
        layout = fdict["layouts"][layout]
    else:
        lname = list(fdict["layouts"].keys())[layout]
        layout = fdict["layouts"][lname]

    x = np.array(layout["coordinates"]["x"], dtype=FC.DTYPE)
    y = np.array(layout["coordinates"]["y"], dtype=FC.DTYPE)
    N = len(x)
    ldata = pd.DataFrame(index=range(N))
    ldata.index.name = "index"
    ldata["x"] = x
    ldata["y"] = y

    tdict = fdict["turbines"]
    pdict = tdict["performance"]

    D = float(tdict["rotor_diameter"])
    H = float(tdict["hub_height"])

    ct_ws = np.array(pdict["Ct_curve"]["Ct_wind_speeds"], dtype=FC.DTYPE)
    ct_data = pd.DataFrame(index=range(len(ct_ws)))
    ct_data["ws"] = ct_ws
    ct_data["ct"] = np.array(pdict["Ct_curve"]["Ct_values"], dtype=FC.DTYPE)

    if "Cp_curve" in pdict:
        cp_ws = np.array(pdict["Cp_curve"]["Cp_wind_speeds"], dtype=FC.DTYPE)
        cp_data = pd.DataFrame(index=range(len(cp_ws)))
        cp_data["ws"] = cp_ws
        cp_data["cp"] = np.array(pdict["Cp_curve"]["Cp_values"], dtype=FC.DTYPE)

        mbook.turbine_types["windio_turbine"] = CpCtFromTwo(
            cp_data, ct_data, col_ws_cp_file="ws", col_cp="cp", D=D, H=H
        )

    elif "power_curve" in pdict:
        P_ws = np.array(pdict["power_curve"]["power_wind_speeds"], dtype=FC.DTYPE)
        P_data = pd.DataFrame(index=range(len(P_ws)))
        P_data["ws"] = P_ws
        P_data["P"] = np.array(pdict["power_curve"]["power_values"], dtype=FC.DTYPE)

        mbook.turbine_types["windio_turbine"] = PCtFromTwo(
            P_data, ct_data, col_ws_P_file="ws", col_P="P", D=D, H=H
        )

    else:
        raise KeyError(f"Missing 'Cp_curve' or 'power_curve' in performance dict, got: {list(pdict.keys())}")

    models = []
    tmnames = [m['model'] for m in tmdict]
    if len(tmdict) and not "turbine_type" in tmnames:
        raise ValueError(f"Missing 'turbine_type' among list of turbine models: {tmnames}")
    elif not len(tmdict):
        models.append("windio_turbine")
    for mdict in tmdict:
        mname = mdict.pop("model")
        if mname != "turbine_type":
            mclass = mdict.pop("class", None)
            if mclass is None and len(mdict):
                raise KeyError(f"Missing parameter 'class' for turbine model '{mname}', expected due to parameters {sorted(list(mdict.keys()))}")
            mbook.get("turbine_models", mname, mclass, **mdict)
        elif len(mdict):
            raise ValueError(f"Turbine model 'turbine_type' does not support parameters: {sorted(list(mdict.keys()))}")
        else:
            mname = "windio_turbine"
        models.append(mname)

    farm = WindFarm(name=fdict["name"])
    add_from_df(farm, ldata, col_x="x", col_y="y", turbine_models=models, **kwargs)

    return farm

def read_anlyses(analyses, mbook, farm, states):
    """
    Reads a WindIO wind farm

    Parameters
    ----------
    analyses: dict
        The analyses sub-dict of the case
    mbook: foxes.ModelBook
        The model book
    farm: foxes.WindFarm
        The wind farm
    states: foxes.states.States
        The states object

    Returns
    -------
    algo: foxes.core.Algorithm
        The algorithm

    """
    fname = analyses["flow_model"]["name"]
    if fname != "foxes":
        raise KeyError(f"Expecting flow model name 'foxes', found '{fname}'")
    
    def _get_models(mtype, wiokey=None, defaults=[]):
        wiok = mtype if wiokey is None else wiokey
        mdicts = analyses.get(wiok, [])
        if not isinstance(mdicts, list):
            mdicts = [mdicts]
        if not len(mdicts):
            return defaults
        models = []
        for mdict in mdicts:
            mname = mdict.pop("model")
            mclass = mdict.pop("class", None)
            if mclass is None and len(mdict):
                raise KeyError(f"Missing parameter 'class' for entry '{mname}' of model type '{mtype}', expected due to parameters {sorted(list(mdict.keys()))}")
            mbook.get(mtype, mname, mclass, **mdict)
            models.append(mname)
        return models

    rotor_model = _get_models("rotor_models", "rotor_model", ["center"])[0]
    wake_models = _get_models("wake_models")
    wake_frame = _get_models("wake_frames", "wake_frame", ["rotor_wd"])[0]
    pwakes = _get_models("partial_wakes", defaults=["auto"])[0]
    farm_controller = _get_models("farm_controllers", "farm_controller", ["basic_ctrl"])[0]

    adict = analyses["algorithm"]
    aclass = adict.pop("class")

    return Algorithm.new(
        aclass, 
        mbook=mbook, 
        farm=farm, 
        states=states, 
        rotor_model=rotor_model,
        wake_models=wake_models, 
        wake_frame=wake_frame,
        partial_wakes_model=pwakes,
        farm_controller=farm_controller,
        **adict,
    )

def read_case(case_data, mbook=None):
    """
    Reads a WindIO case

    Parameters
    ----------
    case_data: dict
        Data from the yaml file
    mbook: foxes.models.ModelBook, optional
        The model book to start from
    site_pars: dict
        Additional arguments for read_site
    farm_pars: dict
        Additional arguments for read_farm
    ana_pars: dict
        Additional arguments for read_analyses

    Returns
    -------
    mbook: foxes.ModelBook
        The model book
    farm: foxes.WindFarm
        The wind farm
    states: foxes.states.States
        The states object
    algo: foxes.core.Algorithm
        The algorithm
    outputs: list of foxes.input.windio.WIOOutput
        The output creating classes

    :group: input.windio

    """
    yml_utils = import_module("windIO.utils.yml_utils", hint="pip install windio")
    case = yml_utils.load_yaml(case_data)
    mbook = ModelBook() if mbook is None else mbook
    adict = case["attributes"]["analyses"]
    olist = adict.pop("outputs", [])

    site_data = case["site"]
    site_pars = adict.pop("site_parameters", {})
    states = read_site(site_data, **site_pars)

    farm_data = case["wind_farm"]
    farm_pars = adict.pop("farm_parameters", {})
    tmdict = adict.pop("turbine_models", {})
    farm = read_farm(farm_data, mbook, tmdict, **farm_pars)

    algo = read_anlyses(adict, mbook, farm, states)

    outputs = []
    for oi, o in enumerate(olist):
        ocls = o.pop("class")
        ofun = o.pop("function")

        oca = o.pop("class_pars", {})
        if oca.pop("algo", False):
            oca["algo"] = algo
        if oca.pop("farm", False):
            oca["farm"] = farm

        o["name"] = o.pop("name", f"output_{oi:02d}")
        o["needs_farm_results"] = oca.pop("farm_results", True)

        outputs.append(WIOOutput(oclass=ocls, ofunction=ofun, ocargs=oca, **o))

    return mbook, farm, states, algo, outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("case_data", help="The case yaml file")
    args = parser.parse_args()

    read_case(args.case_data)
