import time
import argparse
import foxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files_nc",
        help="The input files pattern in the nc folder from iconDream2foxes",
    )
    (
        parser.add_argument(
            "eww_csv",
            help="Path to the EuroWindWakes farm csv file",
        ),
    )
    parser.add_argument(
        "eww_turbines",
        help="Path to the EuroWindWakes turbines data folder",
    )
    parser.add_argument(
        "--farms",
        help="The farms to include (comma-separated)",
        default=["EnBW Windpark Baltic 1"],
        nargs="+",
    )
    parser.add_argument(
        "-e",
        "--engine",
        help="The engine",
        default=None,
    )
    parser.add_argument(
        "-n",
        "--n_cpus",
        help="The number of cpus",
        default=None,
        type=int,
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
    args = parser.parse_args()

    mbook = foxes.ModelBook()

    farm = foxes.WindFarm(
        input_is_lonlat=True,
        utm_zone="from_farm",
    )
    foxes.input.farm_layout.add_from_eww(
        farm,
        args.eww_csv,
        csv_dir=args.eww_turbines,
        filter=[("wind_farm", args.farms)],
        mbook=mbook,
        verbosity=1,
    )

    states = foxes.input.states.ICONStates(args.files_nc)

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        wake_models=["Bastankhah2014_linear_k004"],
        rotor_model="centre",
        mbook=mbook,
        verbosity=1,
    )

    engine = foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
        progress_bar=True,
        verbosity=1,
    )

    with engine:
        time0 = time.time()
        farm_results = algo.calc_farm()
        time1 = time.time()

    print("\nCalc time =", time1 - time0, "\n")
    print(farm_results)
