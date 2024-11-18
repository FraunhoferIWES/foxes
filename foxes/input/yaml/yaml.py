import argparse

from foxes.utils import Dict

from .dict import run_dict


def foxes_yaml():
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
        "-o", "--output_dir", help="Path to the output directory", default=None,
    )
    parser.add_argument(
        "-v", "--verbosity", help="The verbosity level, 0 = silent", type=int, default=None
    )
    args = parser.parse_args()

    v = 1 if args.verbosity is None else args.verbosity
    idata = Dict.from_yaml(args.yml_file, verbosity=v)

    if args.engine is not None:
        epars = dict(
            engine_type=args.engine,
            n_procs=args.n_procs,
            chunk_size_states=args.chunksize_states,
            chunk_size_points=args.chunksize_points,
            verbosity=v,
        )
    else:
        epars = None

    run_dict(
        idata,
        rotor_model=args.rotor,
        partial_wakes=args.pwakes,
        wake_models=args.wakes,
        wake_frame=args.frame,
        engine_pars=epars,
        iterative=args.iterative,
        #output_dir=args.output_dir,
        verbosity=args.verbosity,
    )
