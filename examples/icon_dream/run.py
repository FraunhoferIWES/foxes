import argparse
import foxes


def main(argv=None):
    """Run this tutorial with an explicit argument list or command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nf", "--nofig", action="store_true", help="Do not show figures"
    )
    parser.add_argument(
        "files_nc",
        help="The input files pattern in the nc folder from iconDream2foxes",
    )
    (
        parser.add_argument(
            "eww_db",
            help="Path to the EuroWindWakes farm database csv file",
        ),
    )
    parser.add_argument(
        "eww_pc",
        help="Path to the EuroWindWakes power_curves folder",
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

    # Select chunk outputs by their model-book names.
    parser.add_argument(
        "-R",
        "--results",
        help="The calculation result models",
        nargs="+",
        default=["farm_results"],
    )
    parser.add_argument(
        "-o",
        "--outputs",
        help="The post-processing output models",
        nargs="+",
        default=["state_turbine_table"],
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=2,
        help="0 quiet, 1 summary, 2 reports, 3 library progress",
    )
    args = parser.parse_args(argv)
    if args.verbosity > 0:
        print("Running example: icon_dream")

    mbook = foxes.ModelBook()

    farm = foxes.WindFarm(
        input_is_lonlat=True,
        utm_zone="from_farm",
    )
    foxes.input.farm_layout.add_from_eww(
        farm,
        args.eww_db,
        csv_dir=args.eww_pc,
        filter=[("wind_farm", args.farms)],
        mbook=mbook,
        verbosity=max(args.verbosity - 1, 0),
    )

    states = foxes.input.states.ICONStates(args.files_nc)

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        wake_models=["Bastankhah2014_linear_k004"],
        rotor_model="centre",
        mbook=mbook,
        verbosity=max(args.verbosity - 1, 0),
    )

    engine = foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
        progress_bar=args.verbosity > 2,
        verbosity=max(args.verbosity - 1, 0),
    )

    # Apply figure suppression to both calculation stages.
    for name in [*args.results, *args.outputs]:
        output = mbook.outputs[name]
        output.nofig = args.nofig

    with engine:
        # Preserve every result key, including single-output selections.
        results = algo.calc_farm(outputs=args.results, ret_dict=True)
        if args.verbosity > 1:
            for key, data in results.items():
                print(f"\n{key}:\n{data}")

        # Process each output separately so different point grids remain independent.
        for name in args.outputs:
            output_results = algo.calc_outputs(
                outputs=name, farm_results=results.get("farm_results"), ret_dict=True
            )
            if args.verbosity > 1:
                for key, data in output_results.items():
                    print(f"\n{key}:\n{data}")


if __name__ == "__main__":
    main()
