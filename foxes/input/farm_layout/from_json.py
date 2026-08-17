import json
import numpy as np
from copy import deepcopy
from typing import Any

from foxes.core import Turbine, WindFarm
from foxes.config import get_input_path


def add_from_json(
    farm: WindFarm,
    file_path: str,
    set_farm_name: bool = True,
    verbosity: int = 1,
    **turbine_parameters: Any,
) -> None:
    """
    Add turbimes from a json file.

    Parameters
    ----------
    farm
        The wind farm
    file_path
        Path to the file
    set_farm_name
        Flag for inferring wind farm name from data
    verbosity
        The verbosity level, 0 = silent
    turbine_parameters
        Parameters forwarded to `foxes.core.Turbine`

    :group: input.farm_layout

    """
    fpath = get_input_path(file_path)
    if verbosity:
        print("Reading file", fpath)
    with open(fpath) as f:
        dict = json.load(f)

    keys = list(dict.keys())
    if len(keys) != 1:
        raise KeyError("Only one wind farm supported by foxes at the moment.")

    farm_name = keys[0]
    fdict = dict[farm_name]

    if set_farm_name:
        farm.name = farm_name

    for wt_name, wdict in fdict.items():
        pars = deepcopy(turbine_parameters)
        if "D" in wdict:
            pars["D"] = wdict["D"]
        if "H" in wdict:
            pars["H"] = wdict["H"]
        if "turbine_models" in wdict:
            pars["turbine_models"] = wdict["turbine_models"] + pars.get(
                "turbine_models", []
            )

        wdict = fdict[wt_name]
        farm.add_turbine(
            Turbine(
                xy=np.array([wdict["UTMX"], wdict["UTMY"]]),
                index=wdict.get("id", None),
                name=wt_name,
                **pars,
            ),
            verbosity=verbosity,
        )
