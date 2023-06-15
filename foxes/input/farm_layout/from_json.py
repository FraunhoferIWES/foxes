import json
import numpy as np
from copy import deepcopy

from foxes.core import Turbine


def add_from_json(
    farm, file_path, set_farm_name=True, verbosity=1, **turbine_parameters
):
    """
    Add turbimes from a json file.

    Parameters
    ----------
    farm: foxes.WindFarm
        The wind farm
    file_path: str
        Path to the file
    set_farm_name: bool
        Flag for inferring wind farm name from data
    verbosity: int
        The verbosity level, 0 = silent
    turbine_parameters: dict, optional
        Parameters forwarded to `foxes.core.Turbine`

    :group: input.farm_layout

    """

    if verbosity:
        print("Reading file", file_path)
    with open(file_path) as f:
        dict = json.load(f)

    keys = list(dict.keys())
    if len(keys) != 1:
        raise KeyError("Only one wind farm supported by flappy at the moment.")

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
                **pars
            ),
            verbosity=verbosity,
        )
