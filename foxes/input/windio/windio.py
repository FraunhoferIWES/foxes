import numpy as np
import pandas as pd
from pathlib import Path

from foxes.core import WindFarm, Algorithm
from foxes.models import ModelBook
from foxes.input.states import StatesTable
from foxes.input.farm_layout import add_from_df
from foxes.models.turbine_types import CpCtFromTwo
from foxes.utils import import_module, Dict
from foxes.data import StaticData, WINDIO
import foxes.constants as FC
import foxes.variables as FV

from .read_fields import read_wind_resource_field
from .get_states import get_states
from .read_farm import read_layout, read_turbine_type


def _read_site(wio_site, data, verbosity):
    """ Reads the site information """
    if verbosity > 0:
        print("Reading site")
        print("  Name:", wio_site.pop("name", None))
        print("  Contents:", [k for k in wio_site.keys()])
    
    # ignore boundaries:
    if verbosity > 1:
        print("  Ignoring boundaries")
    
    # read energy_resource:
    energy_resource = Dict(wio_site["energy_resource"], name="energy_resource")
    if verbosity > 1:
        print("  Reading energy_resource")
        print("    Name:", energy_resource.pop("name", None))
        print("    Contents:", [k  for k in energy_resource.keys()])

    # read wind_resource:
    wind_resource = Dict(energy_resource["wind_resource"], name="wind_resource")
    if verbosity > 1:
        print("    Reading wind_resource")
        print("      Name:", wind_resource.pop("name", None))
        print("      Contents:", [k  for k in wind_resource.keys()])

    # read fields
    coords = Dict(name="coords")
    fields = Dict(name="fields")
    dims = Dict(name="dims")
    for n, d in wind_resource.items():
        if verbosity > 1:
            print("        Reading", n)
        read_wind_resource_field(n, d, coords, fields, dims)
    if verbosity > 1:
        print("      Coords:")
        for c, d in coords.items():
            print(f"        {c}: Shape {d.shape}")
        print("      Fields:")
        for f, d in dims.items():
            print(f"        {f}: Dims {d}, shape {fields[f].shape}")

    data["states"] = get_states(coords, fields, dims, verbosity)

def _read_farm(wio_farm, data, verbosity):
    """ Reads the wind farm information """
    if verbosity > 0:
        print("Reading wind farm")
        print("  Name:", wio_farm.pop("name", None))
        print("  Contents:", [k for k in wio_farm.keys()])

    # read turbine type:
    turbines = Dict(wio_farm["turbines"], name="turbines")
    ttype = read_turbine_type(turbines, data, verbosity)

    # read layouts:
    layouts = Dict(wio_farm["layouts"], name="layouts")
    if verbosity > 1:
        print("    Reading layouts")
        print("      Contents:", [k  for k in layouts.keys()])
    for lname, ldict in layouts.items():
        if lname == "initial_layout":
            read_layout(lname, ldict, data, ttype, verbosity)
        elif verbosity > 0:
            print(f"        Ignoring '{lname}'")

def read_windio(windio_yaml, verbosity=2):
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

    if verbosity > 0:
        print(f"Reading windio file {wio_file}")

    yml_utils = import_module("windIO.utils.yml_utils", hint="pip install windio")
    wio = yml_utils.load_yaml(wio_file)

    if verbosity > 0:
        print("  Name:", wio.pop("name", None))
        print("  Contents:", [k for k in wio.keys()])

    data = Dict(mbook=ModelBook())

    _read_site(Dict(wio["site"], name="site"), data, verbosity)
    _read_farm(Dict(wio["wind_farm"], name="farm"), data, verbosity)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="The windio yaml file", default="windio_5turbines_timeseries.yaml")
    args = parser.parse_args()

    read_windio(args.file)
