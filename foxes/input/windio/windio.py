import numpy as np
import pandas as pd

from foxes.core import WindFarm, Algorithm
from foxes.models import ModelBook
from foxes.input.states import StatesTable
from foxes.input.farm_layout import add_from_df
from foxes.models.turbine_types import CpCtFromTwo
from foxes.utils import import_module
import foxes.constants as FC
import foxes.variables as FV


def read_resource(res, fixed_vars={}, **kwargs):
    """
    Reads a WindIO energy resource

    Parameters
    ----------
    res_yaml: str
        Path to the yaml file
    fixed_vars: dict
        Additional fixes variables that do
        not occur in the yaml
    kwargs: dict, optional
        Additional arguments for StatesTable

    Returns
    -------
    states: foxes.states.StatesTable
        The uniform states

    """
    wres = res["wind_resource"]

    wd = np.array(wres["wind_direction"], dtype=FC.DTYPE)
    ws = np.array(wres["wind_speed"], dtype=FC.DTYPE)
    n_wd = len(wd)
    n_ws = len(ws)
    n = n_wd * n_ws

    data = np.zeros((n_wd, n_ws, 2), dtype=FC.DTYPE)
    data[:, :, 0] = wd[:, None]
    data[:, :, 1] = ws[None, :]
    names = ["wind_direction", "wind_speed"]
    sec_prob = None

    def _to_data(v, d, dims):
        nonlocal data, names, sec_prob
        hdata = np.zeros((n_wd, n_ws, 1), dtype=FC.DTYPE)
        if len(dims) == 0:
            hdata[:, :, 0] = FC.DTYPE(d)
        elif len(dims) == 1:
            if dims[0] == "wind_direction":
                hdata[:, :, 0] = np.array(d, dtype=FC.DTYPE)[:, None]
            elif dims[0] == "wind_speed":
                hdata[:, :, 0] = np.array(d, dtype=FC.DTYPE)[None, :]
            else:
                raise ValueError(f"Unknown dimension '{dims[0]}' for data '{v}'")
        elif len(dims) == 2:
            if dims[0] == "wind_direction" and dims[1] == "wind_speed":
                hdata[:, :, 0] = np.array(d, dtype=FC.DTYPE)
            elif dims[1] == "wind_direction" and dims[0] == "wind_speed":
                hdata[:, :, 0] = np.swapaxes(np.array(d, dtype=FC.DTYPE), 0, 1)
            else:
                raise ValueError(f"Cannot handle dims = {dims} for data '{v}'")
        else:
            raise ValueError(
                f"Can not accept more than two dimensions, got {dims} for data '{v}'"
            )
        if v == "sector_probability":
            sec_prob = hdata[:, :, 0].copy()
        else:
            data = np.append(data, hdata, axis=2)
            names.append(v)

    vmap = {
        "wind_direction": FV.WD,
        "wind_speed": FV.WS,
        "turbulence_intensity": FV.TI,
        "air_density": FV.RHO,
        "probability": FV.WEIGHT,
    }

    for v, d in wres.items():
        if (v == "sector_probability" or v in vmap) and isinstance(d, dict):
            _to_data(v, d["data"], d["dims"])
    if sec_prob is not None and "probability" in names:
        data[:, :, names.index("probability")] *= sec_prob

    n_vars = len(names)
    data = data.reshape(n, n_vars)

    data = pd.DataFrame(index=range(n), data=data, columns=names)
    data.index.name = "state"
    data.rename(columns=vmap, inplace=True)

    ovars = {v: v for v in data.columns if v != FV.WEIGHT}
    ovars.update({k: v for k, v in fixed_vars.items() if k not in data.columns})

    return StatesTable(data, output_vars=ovars, fixed_vars=fixed_vars, **kwargs)


def read_site(site, **kwargs):
    """
    Reads a WindIO site

    Parameters
    ----------
    site_yaml: str
        Path to the yaml file
    kwargs: dict, optional
        Additional arguments for read_resource

    Returns
    -------
    states: foxes.states.States
        The states object

    """
    res_yaml = site["energy_resource"]
    states = read_resource(res_yaml, **kwargs)

    return states


def read_farm(fdict, mbook=None, layout=-1, turbine_models=[], **kwargs):
    """
    Reads a WindIO wind farm

    Parameters
    ----------
    farm_yaml: str
        Path to the yaml file
    mbook: foxes.ModelBook, optional
        The model book to start from
    layout: str or int
        The layout choice
    turbine_models: list of str
        Additional turbine models
    kwargs: dict, optional
        Additional parameters for add_from_df()

    Returns
    -------
    mbook: foxes.ModelBook
        The model book
    farm: foxes.WindFarm
        The wind farm

    """
    mbook = ModelBook() if mbook is None else mbook

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

    ct_ws = np.array(pdict["Ct_curve"]["Ct_wind_speeds"], dtype=FC.DTYPE)
    ct_data = pd.DataFrame(index=range(len(ct_ws)))
    ct_data["ws"] = ct_ws
    ct_data["ct"] = np.array(pdict["Ct_curve"]["Ct_values"], dtype=FC.DTYPE)

    cp_ws = np.array(pdict["Cp_curve"]["Cp_wind_speeds"], dtype=FC.DTYPE)
    cp_data = pd.DataFrame(index=range(len(cp_ws)))
    cp_data["ws"] = cp_ws
    cp_data["cp"] = np.array(pdict["Cp_curve"]["Cp_values"], dtype=FC.DTYPE)

    D = float(tdict["rotor_diameter"])
    H = float(tdict["hub_height"])

    mbook.turbine_types["windio_turbine"] = CpCtFromTwo(
        cp_data, ct_data, col_ws_cp_file="ws", col_cp="cp", D=D, H=H
    )

    models = ["windio_turbine"] + turbine_models
    farm = WindFarm(name=fdict["name"])

    add_from_df(farm, ldata, col_x="x", col_y="y", turbine_models=models, **kwargs)

    return mbook, farm


def read_anlyses(
    analyses, mbook, farm, states, keymap={}, algo_type="Downwind", **algo_pars
):
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
    keymap: dict
        Translation from windio to foxes keywords
    algo_type: str
        The default algorithm class name
    algo_pars: dict, optional
        Additional parameters for the algorithm
        constructor

    Returns
    -------
    algo: foxes.core.Algorithm
        The algorithm

    """
    wmodel = analyses["wake_model"]["name"]
    wmodels = [keymap.get(wmodel, wmodel)]

    return Algorithm.new(
        algo_type, mbook, farm, states, wake_models=wmodels, **algo_pars
    )


def read_case(case_yaml, site_pars={}, farm_pars={}, ana_pars={}):
    """
    Reads a WindIO case

    Parameters
    ----------
    case_yaml: str
        Path to the yaml file
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

    :group: input.windio

    """
    yml_utils = import_module("windIO.utils.yml_utils", hint="pip install windio")
    case = yml_utils.load_yaml(case_yaml)

    site_yaml = case["site"]
    states = read_site(site_yaml, **site_pars)

    farm_yaml = case["wind_farm"]
    mbook, farm = read_farm(farm_yaml, **farm_pars)

    attr_dict = case["attributes"]
    algo = read_anlyses(attr_dict["analyses"], mbook, farm, states, **ana_pars)

    return mbook, farm, states, algo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("case_yaml", help="The case yaml file")
    args = parser.parse_args()

    read_case(args.case_yaml)
