import time
import argparse
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV

if __name__ == "__main__":
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
    args = parser.parse_args()

    mbook = foxes.models.ModelBook()
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
    )

    with foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
    ):

        if not args.nofig:
            # fig, axs= plt.subplots(2, 1, figsize=(12,6))
            # foxes.output.FarmLayoutOutput(farm).get_figure(ax=axs[0])

            o = foxes.output.StatesRosePlotOutput(states, point=[0.0, 0.0, 100.0])
            fig = o.get_figure(
                16,
                FV.AMB_WS,
                [0, 3.5, 6, 10, 15, 20],
                figsize=(14.5, 7),
                rect=[0.01, 0.05, 0.45, 0.85],
            )

            ax = plt.Axes(fig, rect=[0.3, 0.1, 0.8, 0.8])
            fig.add_axes(ax)
            foxes.output.FarmLayoutOutput(farm).get_figure(fig=fig, ax=ax)
            plt.show()
            plt.close(fig)

        Algo = (
            foxes.algorithms.Iterative if args.iterative else foxes.algorithms.Downwind
        )
        algo = Algo(
            farm,
            states,
            rotor_model=args.rotor,
            wake_models=args.wakes,
            wake_frame=args.frame,
            partial_wakes=args.pwakes,
            mbook=mbook,
            verbosity=0,
        )

        time0 = time.time()
        farm_results = algo.calc_farm()
        time1 = time.time()

        print("\nFarm results:\n")
        print(farm_results)
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

        o = foxes.output.FarmResultsEval(farm_results)
        o.add_capacity(algo)
        o.add_capacity(algo, ambient=True)
        o.add_efficiency()

        print("\nFarm results:\n")
        print(farm_results)

        # state-turbine results
        farm_df = farm_results.to_dataframe()
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

        # results by turbine
        turbine_results = o.reduce_states(
            {
                FV.AMB_P: "mean",
                FV.P: "mean",
                FV.AMB_CAP: "mean",
                FV.CAP: "mean",
                FV.EFF: "mean",
            }
        )
        turbine_results[FV.AMB_YLD] = o.calc_turbine_yield(
            algo=algo, annual=True, ambient=True
        )
        turbine_results[FV.YLD] = o.calc_turbine_yield(algo=algo, annual=True)
        print("\nResults by turbine:\n")
        print(turbine_results)

        # power results
        P0 = o.calc_mean_farm_power(ambient=True)
        P = o.calc_mean_farm_power()
        print(f"\nFarm power        : {P/1000:.1f} MW")
        print(f"Farm ambient power: {P0/1000:.1f} MW")
        print(f"Farm efficiency   : {o.calc_farm_efficiency()*100:.2f} %")
        print(f"Annual farm yield : {turbine_results[FV.YLD].sum():.2f} GWh")
