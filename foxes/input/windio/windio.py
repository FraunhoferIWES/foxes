import numpy as np
import pandas as pd
from pathlib import Path

from foxes.core import WindFarm, Algorithm
from foxes.models import ModelBook
from foxes.input.states import StatesTable
from foxes.input.farm_layout import add_from_df
from foxes.models.turbine_types import CpCtFromTwo
from foxes.utils import import_module
from foxes.data import StaticData, WINDIO
import foxes.constants as FC
import foxes.variables as FV





def read_windio(windio_yaml, verbosity=1):
    """
    Reads a WindIO case

    Parameters
    ----------
    windio_yaml: str
        Path to the windio yaml file
    verbosity: int
        The verbosity level, 0 = silent

    Returns
    -------
    algo: foxes.core.Algorithm
        The algorithm

    :group: input.windio

    """

    wio_file = Path(windio_yaml)
    if not wio_file.is_file():
        wio_file = StaticData().get_file_path(
            WINDIO, wio_file, check_raw=False
        ) 

    yml_utils = import_module("windIO.utils.yml_utils", hint="pip install windio")
    wio = yml_utils.load_yaml(wio_file)

    print(wio)




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="The windio yaml file", default="windio_5turbines_timeseries.yaml")
    args = parser.parse_args()

    read_windio(args.file)
