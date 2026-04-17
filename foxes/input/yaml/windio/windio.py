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


def windio_file2dict(yml_file, verbosity=1):
    """
    Read windio yaml file and translate to foxes input data dictionary

    Parameters
    ----------
    yml_file: pathlib.Path or str
        The windio yaml file
    verbosity: int
        The verbosity level, 0 = silent

    Returns
    -------
    wio_dict: foxes.utils.Dict
        The windio data dictionary

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

    return Dict(windio.load_yaml(wio_file), _name="windio")


def read_windio_dict(wio_dict, verbosity=1, **algo_kwargs):
    """
    Translate windio data to foxes input data

    Parameters
    ----------
    wio_dict: foxes.utils.Dict
        The windio data
    ret_algo: bool
        Whether to return the algorithm object, otherwise
        return its parameters
    verbosity: int
        The verbosity level, 0 = silent
    algo_kwargs: dict, optional
        Additional keyword arguments for the algorithm

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

    idict["algorithm"].update(algo_kwargs)
    if verbosity > 1:
        print("\nFinal input dictionary:\n\n", idict, "\n")
    algo = Algorithm.new(
        farm=farm, states=states, mbook=mbook, **idict.pop_item("algorithm")
    )

    odir = None
    if "model_outputs_specification" in wio_attrs:
        outputs = wio_attrs["model_outputs_specification"]
        odir = read_outputs(outputs, idict, algo, verbosity=verbosity)

    return idict, algo, odir


def read_windio_file(yml_file, ret_wio=False, verbosity=1, **algo_kwargs):
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
    algo_kwargs: dict, optional
        Additional keyword arguments for the algorithm

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
    wio = windio_file2dict(yml_file, verbosity)
    idict, algo, odir = read_windio_dict(wio, verbosity=verbosity, **algo_kwargs)

    if ret_wio:
        return idict, algo, odir, wio
    else:
        return idict, algo, odir


def foxes_windio(
    yml_file,
    output_dir=None,
    rotor=None,
    pwakes=None,
    wakes=None,
    frame=None,
    engine=None,
    n_procs=None,
    chunksize_states=None,
    chunksize_points=None,
    iterative=False,
    nofig=False,
    verbosity=1,
):
    """Run foxes from windio yaml file input

    Parameters
    ----------
    yml_file: str or Path
        The yaml file path
    output_dir: str or Path, optional
        The output directory, default: None (same as input file)
    rotor: str, optional
        The rotor model, default: None (use the one from the yaml file)
    pwakes: list of str, optional
        The partial wakes models, default: None (use the ones from the yaml file)
    wakes: list of str, optional
        The wake models, default: None (use the ones from the yaml file)
    frame: str, optional
        The wake frame, default: None (use the one from the yaml file)
    engine: str, optional
        The engine, default: None (use the one from the yaml file)
    n_procs: int, optional
        The number of processes, default: None (use the one from the yaml file)
    chunksize_states: int, optional
        The chunk size for states, default: None (use the one from the yaml file)
    chunksize_points: int, optional
        The chunk size for points, default: None (use the one from the yaml file)
    iterative: bool, optional
        Use iterative algorithm, default: False
    nofig: bool, optional
        Do not show figures, default: False
    verbosity: int, optional
        The verbosity level, 0 = silent, default: 1

    Returns
    -------
    farm_results: xarray.Dataset, optional
        The farm results
    point_results: xarray.Dataset, optional
        The point results
    outputs: list of tuple
        For each output enty, a tuple (dict, results),
        where results is a list that represents one
        entry per function call


    """

    if (
        engine is not None
        or n_procs is not None
        or chunksize_states is not None
        or chunksize_points is not None
    ):
        epars = dict(
            engine_type=engine,
            n_procs=n_procs,
            chunk_size_states=chunksize_states,
            chunk_size_points=chunksize_points,
            verbosity=verbosity,
        )
    else:
        epars = None

    wio_file = Path(yml_file)
    idict, algo, odir = read_windio_file(wio_file, verbosity=verbosity)

    if output_dir is not None:
        odir = output_dir

    return run_dict(
        idict,
        algo=algo,
        rotor_model=rotor,
        partial_wakes=pwakes,
        wake_models=wakes,
        wake_frame=frame,
        engine_pars=epars,
        iterative=iterative,
        input_dir=wio_file.parent,
        output_dir=odir,
        verbosity=verbosity,
    )


def main():
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

    foxes_windio(
        yml_file=args.yml_file,
        output_dir=args.output_dir,
        rotor=args.rotor,
        pwakes=args.pwakes,
        wakes=args.wakes,
        frame=args.frame,
        engine=args.engine,
        n_procs=args.n_procs,
        chunksize_states=args.chunksize_states,
        chunksize_points=args.chunksize_points,
        iterative=args.iterative,
        nofig=args.nofig,
        verbosity=args.verbosity,
    )


if __name__ == "__main__":
    main()
