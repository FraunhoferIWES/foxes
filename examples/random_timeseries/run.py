import time
import argparse
import matplotlib.pyplot as plt

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
        print("Running example: random_timeseries")

    mbook = foxes.models.ModelBook()
    mbook.outputs["farm_results"].fdata_variables.extend(
        [FV.AMB_CT, FV.AMB_WD, FV.CT, FV.EFF]
    )

    if args.nofig:
        for name in args.outputs:
            output = mbook.outputs[name]
            if hasattr(output, "show"):
                output.show = False
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
    if not args.nofig:
        fig = plt.figure(figsize=(14.5, 7))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, polar=True)
        foxes.algorithms.Downwind(
            farm,
            foxes.input.states.SingleStateStates(ws=8, wd=270, ti=0.08, rho=1.225),
            wake_models=[],
            mbook=mbook,
            verbosity=max(args.verbosity - 1, 0),
        ).calc_outputs(
            outputs=foxes.models.outputs.FarmLayoutOutput(
                operation="get_figure", parameters=dict(fig=fig, ax=ax1)
            )
        )

        with engine:
            foxes.algorithms.Downwind(
                farm,
                states,
                mbook=mbook,
                wake_models=[],
                verbosity=max(args.verbosity - 1, 0),
            ).calc_outputs(
                outputs=foxes.models.outputs.StatesRosePlotOutput(
                    point=[0.0, 0.0, 100.0],
                    plot=True,
                    plot_pars=dict(fig=fig, ax=ax2, freq_delta=2),
                )
            )[0]

        plt.show()
        plt.close(fig)

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
    with engine:
        time0 = time.time()
        # Request per-state metrics through output selection.
        farm_results = algo.calc_farm(outputs=args.outputs)
        time1 = time.time()

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
    required = {
        FV.AMB_CT,
        FV.AMB_P,
        FV.AMB_REWS,
        FV.AMB_TI,
        FV.AMB_WD,
        FV.CT,
        FV.EFF,
        FV.P,
        FV.REWS,
        FV.TI,
        FV.WD,
        FV.X,
    }
    if farm_results is None or not required.issubset(farm_results.data_vars):
        print(
            "Skipping farm reporting: selected outputs do not collect the required fields:",
            sorted(required),
        )
        return

    if args.verbosity > 1:
        print(
            farm_results.to_dataframe()[
                [
                    FV.AMB_WD,
                    FV.AMB_TI,
                    FV.AMB_REWS,
                    FV.AMB_CT,
                    FV.AMB_P,
                    FV.TI,
                    FV.REWS,
                    FV.CT,
                    FV.P,
                ]
            ]
        )
        print("\nCalc time =", time1 - time0, "\n")

    # state-turbine results
    farm_df = farm_results.to_dataframe()
    if args.verbosity > 1:
        print("\nFarm results data:\n")
        print(
            farm_df[
                [
                    FV.X,
                    FV.WD,
                    FV.AMB_REWS,
                    FV.REWS,
                    FV.AMB_TI,
                    FV.TI,
                    FV.AMB_P,
                    FV.P,
                    FV.EFF,
                ]
            ]
        )
        print()


if __name__ == "__main__":
    main()
