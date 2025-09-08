import argparse
from pathlib import Path

from foxes.core import Algorithm
from foxes.models import ModelBook
from foxes.utils import import_module, Dict

from .read_site import read_site
from .read_farm import read_farm
from .read_attributes import read_attributes
from .read_outputs import read_outputs
from ..dict import run_dict


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
    idict: foxes.utils.Dict or dict
        The foxes input data dictionary
    algo: foxes.core.Algorithm
        The algorithm
    odir: pathlib.Path
        The output directory

    :group: input.yaml.windio

    """

    def _print(*args, level=1, **kwargs):
        if verbosity >= level:
            print(*args, **kwargs)

    if not isinstance(wio_dict, Dict):
        tmp = Dict(_name="windio")
        for k, d in wio_dict.items():
            tmp[k] = d
        wio_dict = tmp

    _print("Reading windio data")
    _print("  Name:", wio_dict.pop_item("name", None))
    _print("  Contents:", [k for k in wio_dict.keys()])

    idict = Dict(
        wind_farm=Dict(_name="wio2fxs.farm"),
        algorithm=Dict(
            algo_type="Downwind",
            wake_models=[],
            farm_controller="farm_cntrl",
            _name="wio2fxs.algorithm",
            verbosity=verbosity - 3,
        ),
        calc_farm=Dict(run=True, _name="wio2fxs.calc_farm"),
        _name="wio2fxs",
    )

    mbook = ModelBook()
    states = read_site(wio_dict, verbosity)
    farm = read_farm(wio_dict, mbook, verbosity)

    wio_attrs = wio_dict["attributes"]
    read_attributes(wio_attrs, idict, mbook, verbosity=verbosity)

    # special case WeibullPointCloud:
    if (
        type(states).__name__ == "WeibullPointCloud"
        and idict["algorithm"]["rotor_model"] == "centre"
    ):
        _print(
            "Found WeibullPointCloud states, changing rotor model from 'centre' to 'direct_mdata'",
            level=3,
        )
        idict["algorithm"]["rotor_model"] = "direct_mdata"

    algo = Algorithm.new(
        farm=farm, states=states, mbook=mbook, **idict.pop_item("algorithm")
    )

    odir = None
    if "model_outputs_specification" in wio_attrs:
        outputs = wio_attrs["model_outputs_specification"]
        odir = read_outputs(outputs, idict, algo, verbosity=verbosity)

    return idict, algo, odir


def read_windio_file(yml_file, ret_wio=False, verbosity=1):
    """
    Read windio yaml file and translate to foxes input data

    Parameters
    ----------
    yml_file: pathlib.Path or str
        The windio yaml file
    ret_wio: bool
        Whether to return the windio data dictionary as well
    verbosity: int
        The verbosity level, 0 = silent

    Returns
    -------
    idict: foxes.utils.Dict or dict
        The foxes input data dictionary
    algo: foxes.core.Algorithm
        The algorithm
    odir: pathlib.Path
        The output directory

    :group: input.yaml.windio

    """

    wio_file = Path(yml_file)
    if verbosity > 0:
        print(f"Reading windio file {wio_file}")

    windio = import_module(
        "windIO",
        pip_hint="pip install git+https://github.com/EUFLOW/windIO@master",
        conda_hint="",
    )

    wio = Dict(windio.load_yaml(wio_file), _name="windio")
    idict, algo, odir = read_windio(wio, verbosity=verbosity)

    if ret_wio:
        return idict, algo, odir, wio
    else:
        return idict, algo, odir


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
    parser.add_argument("-o", "--output_dir", help="The output directory", default=None)
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
        default=1,
    )
    args = parser.parse_args()

    def _print(*ags, level=1, **kwargs):
        if args.verbosity >= level:
            print(*ags, **kwargs)

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
            verbosity=args.verbosity,
        )
    else:
        epars = None

    wio_file = Path(args.yml_file)
    idict, algo, odir = read_windio_file(wio_file, verbosity=args.verbosity)

    if args.output_dir is not None:
        odir = args.odir

    run_dict(
        idict,
        algo=algo,
        rotor_model=args.rotor,
        partial_wakes=args.pwakes,
        wake_models=args.wakes,
        wake_frame=args.frame,
        engine_pars=epars,
        iterative=args.iterative,
        input_dir=wio_file.parent,
        output_dir=odir,
        verbosity=args.verbosity,
    )
