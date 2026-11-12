import argparse

import foxes
import foxes.variables as FV


def main(argv=None):
    """Run this tutorial with an explicit argument list or command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "n_turbines",
        help="The number of wind turbines",
        type=int,
    )
    parser.add_argument(
        "n_times",
        help="The number of states in the time series",
        type=int,
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="The random seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes models", default="centre", nargs="+"
    )
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["Bastankhah2014_linear_k004"],
        nargs="+",
    )
    parser.add_argument("-f", "--frame", help="The wake frame", default="rotor_wd")
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
        "-it", "--iterative", help="Use iterative algorithm", action="store_true"
    )
    parser.add_argument(
        "-nf", "--nofig", help="Do not show figures", action="store_true"
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
        default=["farm_layout", "rose_plot"],
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
        print("Running example: random_timeseries")

    mbook = foxes.models.ModelBook()
    mbook.outputs["farm_results"].fdata_variables.extend(
        [FV.AMB_CT, FV.AMB_WD, FV.CT, FV.EFF]
    )

    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    sdata = foxes.input.states.create.random_timseries_data(
        args.n_times, seed=args.seed
    )
    states = foxes.input.states.Timeseries(
        data_source=sdata,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.RHO: 1.225, FV.TI: 0.04},
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_random(
        farm,
        args.n_turbines,
        min_dist=500,
        turbine_models=args.tmodels + [ttype.name],
        seed=args.seed,
        verbosity=max(args.verbosity - 1, 0),
    )

    engine = foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
        verbosity=max(args.verbosity - 1, 0),
    )

    Algo = foxes.algorithms.Iterative if args.iterative else foxes.algorithms.Downwind
    algo = Algo(
        farm,
        states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame=args.frame,
        partial_wakes=args.pwakes,
        mbook=mbook,
        verbosity=max(args.verbosity - 1, 0),
    )
    mbook.outputs["rose_plot"].wd_var = FV.WD
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
