import argparse

import foxes
import foxes.variables as FV


def main(argv=None):
    """Run this tutorial with an explicit argument list or command-line options."""
    # define arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-P",
        "--P_percent",
        help="Power percent choice, applied to all states",
        type=float,
        default=55.0,
    )
    parser.add_argument(
        "-s",
        "--states",
        help="The states input file (path or static)",
        default="wind_rose_bremen.csv",
    )
    parser.add_argument(
        "-l",
        "--layout",
        help="The wind farm layout file (path or static)",
        default="test_farm_67.csv",
    )
    parser.add_argument(
        "-D",
        help="The rotor diameter",
        default=126.0,
        type=float,
    )
    parser.add_argument(
        "-H",
        help="The hub height",
        default=90.0,
        type=float,
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes models", default=None, nargs="+"
    )
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["Bastankhah2014_linear_lim_k004"],
        nargs="+",
    )
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=[], nargs="+"
    )
    parser.add_argument("-e", "--engine", help="The engine", default="default")
    parser.add_argument(
        "-n", "--n_cpus", help="The number of cpus", default=None, type=int
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
        "-nf", "--nofig", help="Do not show figures", action="store_true"
    )

    # Select chunk outputs by their model-book names.
    parser.add_argument(
        "-o", "--outputs", help="The output models", nargs="+", default=["farm_results"]
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
        print("Running example: states_lookup_table")

    mbook = foxes.models.ModelBook()
    mbook.outputs["farm_results"].fdata_variables.append(FV.CT)

    if args.nofig:
        for name in args.outputs:
            output = mbook.outputs[name]
            if hasattr(output, "show"):
                output.show = False
    mbook.turbine_types["ttypeDH"] = foxes.models.turbine_types.NullType(
        D=args.D, H=args.H
    )

    mbook.turbine_models["lookup"] = foxes.models.turbine_models.LookupTable(
        "curtail_to_power.csv",
        input_vars=[FV.REWS, "P_percent"],
        output_vars=[FV.P, FV.CT],
        varmap={
            FV.REWS: "wind",
            "P_percent": "powerPercent",
            FV.P: "GenPower",
            FV.CT: "ct",
        },
        P_percent=args.P_percent,
    )

    states = foxes.input.states.StatesTable(
        data_source=args.states,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.WEIGHT: "weight"},
        fixed_vars={FV.RHO: 1.225, FV.TI: 0.05},
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_from_file(
        farm,
        args.layout,
        col_x="x",
        col_y="y",
        col_H="H",
        turbine_models=["ttypeDH", "lookup"] + args.tmodels,
        verbosity=max(args.verbosity - 1, 0),
    )

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame="rotor_wd",
        partial_wakes=args.pwakes,
        mbook=mbook,
        verbosity=max(args.verbosity - 1, 0),
    )

    engine = foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
        verbosity=max(args.verbosity - 1, 0),
    )

    with engine:
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
    required = {FV.AMB_P, FV.AMB_REWS, FV.CT, FV.H, FV.P, FV.REWS, FV.WD, FV.WEIGHT}
    if farm_results is None or not required.issubset(farm_results.data_vars):
        print(
            "Skipping farm reporting: selected outputs do not collect the required fields:",
            sorted(required),
        )
        return

    fr = farm_results.to_dataframe()
    if args.verbosity > 1:
        print(fr[[FV.WD, FV.H, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P, FV.CT, FV.WEIGHT]])


if __name__ == "__main__":
    main()
