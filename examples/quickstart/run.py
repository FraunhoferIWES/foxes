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
        "-o", "--outputs", help="The output models", nargs="+", default=["farm_results"]
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
    if args.nofig:
        for name in args.outputs:
            output = mbook.outputs[name]
            if hasattr(output, "show"):
                output.show = False
    algo = foxes.algorithms.Downwind(
        farm,
        states,
        ["Jensen_linear_k007"],
        mbook=mbook,
        verbosity=max(args.verbosity - 1, 0),
    )
    with foxes.Engine.new(args.engine, verbosity=max(args.verbosity - 1, 0)):
        farm_results = algo.calc_farm(outputs=args.outputs)

    # Evaluate collected farm data only; writer-only outputs are already complete.
    if isinstance(farm_results, dict):
        farm_results = farm_results.get("farm_results")
    elif not any(
        isinstance(mbook.outputs[name], foxes.models.outputs.FarmResultsOutput)
        for name in args.outputs
    ):
        farm_results = None
    if farm_results is not None and args.verbosity > 1:
        print("\nFarm results:\n", farm_results)
    if farm_results is None:
        print("Skipping farm reporting: no farm-results output was selected.")
        return


if __name__ == "__main__":
    main()
