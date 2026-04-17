import argparse
from pathlib import Path

from foxes.utils import Dict

from .dict import run_dict


def foxes_yaml(
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
    """
    Run foxes from yaml file input

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

    :group: input.yaml

    """

    v = 1 if verbosity is None else verbosity
    fpath = Path(yml_file)
    idata = Dict.from_yaml(fpath, verbosity=v)

    if (
        engine is not None
        or n_procs is not None
        or chunksize_states is not None
        or chunksize_points is not None
    ):
        epars = dict(
            engine_type=engine if engine is not None else "default",
            n_procs=n_procs,
            chunk_size_states=chunksize_states,
            chunk_size_points=chunksize_points,
            verbosity=v,
        )
    else:
        epars = None

    return run_dict(
        idata,
        rotor_model=rotor,
        partial_wakes=pwakes,
        wake_models=wakes,
        wake_frame=frame,
        engine_pars=epars,
        iterative=iterative,
        input_dir=fpath.parent,
        output_dir=output_dir,
        nofig=nofig,
        verbosity=verbosity,
    )


def main():
    """
    Command line tool for running foxes from yaml file input.

    Examples
    --------
    >>> foxes_yaml input.yaml

    :group: input.yaml

    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "yml_file",
        help="The input yaml file",
    )
    parser.add_argument("-o", "--output_dir", help="The output directory", default=None)
    parser.add_argument("-r", "--rotor", help="The rotor model", default=None)
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes models", default=None, nargs="+"
    )
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=None,
        nargs="+",
    )
    parser.add_argument("-f", "--frame", help="The wake frame", default=None)
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
        default=None,
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

    foxes_yaml(
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
