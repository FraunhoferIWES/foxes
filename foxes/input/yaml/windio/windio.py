import argparse
from pathlib import Path

from foxes.core import WindFarm
from foxes.models import ModelBook
from foxes.utils import import_module, Dict

from .read_fields import read_wind_resource_field
from .get_states import get_states
from .read_farm import read_layout, read_turbine_types
from .read_attributes import read_attributes
from ..dict import run_dict


def _read_site(wio_dict, verbosity):
    """Reads the site information"""

    def _print(*args, level=1, **kwargs):
        if verbosity >= level:
            print(*args, **kwargs)

    wio_site = Dict(wio_dict["site"], name=wio_dict.name + ".site")
    _print("Reading site")
    _print("  Name:", wio_site.pop("name", None))
    _print("  Contents:", [k for k in wio_site.keys()])
    _print("  Ignoring boundaries", level=2)

    # read energy_resource:
    energy_resource = Dict(
        wio_site["energy_resource"], name=wio_site.name + ".energy_resource"
    )
    _print("  Reading energy_resource", level=2)
    _print("    Name:", energy_resource.pop("name", None), level=2)
    _print("    Contents:", [k for k in energy_resource.keys()], level=2)

    # read wind_resource:
    wind_resource = Dict(
        energy_resource["wind_resource"], name=energy_resource.name + ".wind_resource"
    )
    _print("    Reading wind_resource", level=3)
    _print("      Name:", wind_resource.pop("name", None), level=3)
    _print("      Contents:", [k for k in wind_resource.keys()], level=3)

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

    return get_states(coords, fields, dims, verbosity)


def _read_farm(wio_dict, mbook, verbosity):
    """Reads the wind farm information"""
    wio_farm = Dict(wio_dict["wind_farm"], name=wio_dict.name + ".wind_farm")
    if verbosity > 1:
        print("Reading wind farm")
        print("  Name:", wio_farm.pop("name", None))
        print("  Contents:", [k for k in wio_farm.keys()])

    # find REWS exponents:
    try:
        rotor_averaging = wio_dict["attributes"]["analysis"]["rotor_averaging"]
        ws_exp_P = rotor_averaging["wind_speed_exponent_for_power"]
        ws_exp_ct = rotor_averaging["wind_speed_exponent_for_ct"]
    except KeyError:
        ws_exp_P = 1
        ws_exp_ct = 1

    # read turbine type:
    ttypes = read_turbine_types(wio_farm, mbook, ws_exp_P, ws_exp_ct, verbosity)

    # read layouts and create wind farm:
    farm = WindFarm()
    wfarm = wio_farm["layouts"]
    if isinstance(wfarm, dict):
        layouts = Dict(wfarm, name=wio_farm.name + ".layouts")
    else:
        layouts = {str(i): l for i, l in enumerate(wfarm)}
        layouts = Dict(layouts, name=wio_farm.name + ".layouts")
    if verbosity > 2:
        print("    Reading layouts")
        print("      Contents:", [k for k in layouts.keys()])
    for lname, ldict in layouts.items():
        read_layout(lname, ldict, farm, ttypes, verbosity)

    return farm


def read_windio(wio_dict, verbosity=1):
    """
    Translate windio data to foxes input data

    Parameters
    ----------
    wio_dict: foxes.utils.Dict
        The windio data
    verbosity: int
        The verbosity level, 0 = silent

    Returns
    -------
    idict: foxes.utils.Dict
        The foxes input data dictionary
    states: foxes.core.States
        The states object
    farm: foxes.core.WindFarm
        The wind farm
    mbook: foxes.models.ModelBook
        The model book

    :group: input.yaml.windio

    """

    def _print(*args, level=1, **kwargs):
        if verbosity >= level:
            print(*args, **kwargs)

    _print(f"Reading windio data")
    _print("  Name:", wio_dict.pop("name", None))
    _print("  Contents:", [k for k in wio_dict.keys()])

    idict = Dict(
        wind_farm=Dict(name="wio2fxs.farm"),
        algorithm=Dict(
            algo_type="Downwind",
            wake_models=[],
            name="wio2fxs.algorithm",
            verbosity=verbosity - 3,
        ),
        calc_farm=Dict(run=True, name="wio2fxs.calc_farm"),
        outputs=Dict(name="wio2fxs.outputs"),
        name="wio2fxs",
    )

    mbook = ModelBook()
    states = _read_site(wio_dict, verbosity)
    farm = _read_farm(wio_dict, mbook, verbosity)

    odir = read_attributes(wio_dict, idict, mbook, verbosity=verbosity)

    return idict, states, farm, mbook, odir


def foxes_windio():
    """
    Command line tool for running foxes from windio yaml file input.

    Examples
    --------
    >>> foxes_windio input.yaml

    :group: input.yaml.windio

    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "yml_file",
        help="The windio yaml file",
    )
    parser.add_argument("-o", "--out_dir", help="The output directory", default=None)
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes models", default="centre", nargs="+"
    )
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["Jensen_linear_k007"],
        nargs="+",
    )
    parser.add_argument("-f", "--frame", help="The wake frame", default="rotor_wd")
    parser.add_argument("-e", "--engine", help="The engine", default=None)
    parser.add_argument(
        "-n", "--n_procs", help="The number of processes", default=None, type=int
    )
    parser.add_argument(
        "-c",
        "--chunksize_states",
        help="The chunk size for states",
        default=None,
        type=int,
    )
    parser.add_argument(
        "-C",
        "--chunksize_points",
        help="The chunk size for points",
        default=5000,
        type=int,
    )
    parser.add_argument(
        "-it", "--iterative", help="Use iterative algorithm", action="store_true"
    )
    parser.add_argument(
        "-nf", "--nofig", help="Do not show figures", action="store_true"
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        help="The verbosity level, 0 = silent",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    v = 1 if args.verbosity is None else args.verbosity

    def _print(*args, level=1, **kwargs):
        if v >= level:
            print(*args, **kwargs)

    if (
        args.engine is not None
        or args.n_procs is not None
        or args.chunksize_states is not None
        or args.chunksize_points is not None
    ):
        epars = dict(
            engine_type=args.engine,
            n_procs=args.n_procs,
            chunk_size_states=args.chunksize_states,
            chunk_size_points=args.chunksize_points,
            verbosity=v,
        )
    else:
        epars = None

    wio_file = Path(args.yml_file)
    _print(f"Reading windio file {wio_file}")
    yml_utils = import_module(
        "windIO.utils.yml_utils",
        hint="pip install git+https://github.com/kilojoules/windIO@master#egg=windIO",
    )

    wio = Dict(yml_utils.load_yaml(wio_file), name="windio")
    idict, states, farm, mbook, odir = read_windio(wio, verbosity=v)

    if args.out_dir is not None:
        odir = args.odir

    run_dict(
        idict,
        farm=farm,
        states=states,
        mbook=mbook,
        rotor_model=args.rotor,
        partial_wakes=args.pwakes,
        wake_models=args.wakes,
        wake_frame=args.frame,
        engine_pars=epars,
        iterative=args.iterative,
        work_dir=wio_file.parent,
        out_dir=odir,
        verbosity=args.verbosity,
    )
