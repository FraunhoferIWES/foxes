import argparse

import foxes


def main(argv=None):
    """Run this tutorial with an explicit argument list or command-line options."""
    # Select chunk outputs by their model-book names.
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", default=None)
    parser.add_argument("-s", "--states", default="timeseries_3000.csv.gz")
    parser.add_argument("-l", "--layout", default="test_farm_67.csv")
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
        "-nf", "--nofig", help="Do not show figures", action="store_true"
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
        print("Running example: quickstart")

    states = foxes.input.states.Timeseries(args.states, ["WS", "WD", "TI", "RHO"])

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_from_file(
        farm,
        args.layout,
        turbine_models=["NREL5MW"],
        verbosity=max(args.verbosity - 1, 0),
    )

    mbook = foxes.ModelBook()
    algo = foxes.algorithms.Downwind(
        farm,
        states,
        ["Jensen_linear_k007"],
        mbook=mbook,
        verbosity=max(args.verbosity - 1, 0),
    )
    # Apply figure suppression to both calculation stages.
    for name in [*args.results, *args.outputs]:
        output = mbook.outputs[name]
        output.nofig = args.nofig

    with foxes.Engine.new(args.engine, verbosity=max(args.verbosity - 1, 0)):
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
