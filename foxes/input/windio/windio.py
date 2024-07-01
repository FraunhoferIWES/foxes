import numpy as np
import pandas as pd
from pathlib import Path

from foxes.core import WindFarm
from foxes.models import ModelBook
from foxes.utils import import_module, Dict
from foxes.data import StaticData, WINDIO

from .read_fields import read_wind_resource_field
from .get_states import get_states
from .read_farm import read_layout, read_turbine_type
from .read_attributes import read_attributes
from .runner import WindioRunner


def _read_site(wio, algo_dict, verbosity):
    """Reads the site information"""
    wio_site = Dict(wio["site"], name="site")
    if verbosity > 1:
        print("Reading site")
        print("  Name:", wio_site.pop("name", None))
        print("  Contents:", [k for k in wio_site.keys()])

    # ignore boundaries:
    if verbosity > 2:
        print("  Ignoring boundaries")

    # read energy_resource:
    energy_resource = Dict(wio_site["energy_resource"], name="energy_resource")
    if verbosity > 2:
        print("  Reading energy_resource")
        print("    Name:", energy_resource.pop("name", None))
        print("    Contents:", [k for k in energy_resource.keys()])

    # read wind_resource:
    wind_resource = Dict(energy_resource["wind_resource"], name="wind_resource")
    if verbosity > 2:
        print("    Reading wind_resource")
        print("      Name:", wind_resource.pop("name", None))
        print("      Contents:", [k for k in wind_resource.keys()])

    # read fields
    coords = Dict(name="coords")
    fields = Dict(name="fields")
    dims = Dict(name="dims")
    for n, d in wind_resource.items():
        read_wind_resource_field(n, d, coords, fields, dims, verbosity)
    if verbosity > 2:
        print("      Coords:")
        for c, d in coords.items():
            print(f"        {c}: Shape {d.shape}")
        print("      Fields:")
        for f, d in dims.items():
            if len(d):
                print(f"        {f}: Dims {d}, shape {fields[f].shape}")
            else:
                print(f"        {f} = {fields[f]}")

    algo_dict["states"] = get_states(coords, fields, dims, verbosity)


def _read_farm(wio, algo_dict, verbosity):
    """Reads the wind farm information"""
    wio_farm = Dict(wio["wind_farm"], name="wind_farm")
    if verbosity > 1:
        print("Reading wind farm")
        print("  Name:", wio_farm.pop("name", None))
        print("  Contents:", [k for k in wio_farm.keys()])

    # find REWS exponents:
    try:
        rotor_averaging = wio["attributes"]["analysis"]["rotor_averaging"]
        ws_exp_P = rotor_averaging["wind_speed_exponent_for_power"]
        ws_exp_ct = rotor_averaging["wind_speed_exponent_for_ct"]
    except KeyError:
        ws_exp_P = 1
        ws_exp_ct = 1

    # read turbine type:
    turbines = Dict(wio_farm["turbines"], name="turbines")
    ttype = read_turbine_type(turbines, algo_dict, ws_exp_P, ws_exp_ct, verbosity)

    # read layouts:
    layouts = Dict(wio_farm["layouts"], name="layouts")
    if verbosity > 2:
        print("    Reading layouts")
        print("      Contents:", [k for k in layouts.keys()])
    for lname, ldict in layouts.items():
        read_layout(lname, ldict, algo_dict, ttype, verbosity)


def read_windio(windio_yaml, verbosity=1):
    """
    Reads a complete WindIO case.

    This is the main entry point for windio case
    calculations.

    Parameters
    ----------
    windio_yaml: str
        Path to the windio yaml file
    verbosity: int
        The verbosity level, 0 = silent

    Returns
    -------
    runner: foxes.input.windio.WindioRunner
        The windio runner, call its run function
        for the complete exection

    :group: input.windio

    """

    wio_file = Path(windio_yaml)
    if not wio_file.is_file():
        wio_file = StaticData().get_file_path(WINDIO, wio_file, check_raw=False)

    if verbosity > 0:
        print(f"Reading windio file {wio_file}")

    yml_utils = import_module("windIO.utils.yml_utils", hint="pip install windio")
    wio = yml_utils.load_yaml(wio_file)

    if verbosity > 1:
        print("  Name:", wio.pop("name", None))
        print("  Contents:", [k for k in wio.keys()])

    algo_dict = Dict(
        algo_type="Downwind",
        mbook=ModelBook(),
        farm=WindFarm(),
        wake_models=[],
        verbosity=verbosity-3,
    )

    _read_site(wio, algo_dict, verbosity)
    _read_farm(wio, algo_dict, verbosity)
    
    out_dicts, odir = read_attributes(
        wio,
        algo_dict,
        verbosity,
    )

    if verbosity > 1:
        print("Creating windio runner")
    runner = WindioRunner(
                algo_dict, 
                output_dir=odir,
                output_dicts=out_dicts, 
                wio_input_data=wio, 
                verbosity=verbosity
            )

    return runner


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        help="The windio yaml file",
        default="windio_5turbines_timeseries.yaml",
    )
    args = parser.parse_args()

    runner = read_windio(args.file)

    runner.run()
