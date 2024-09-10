import time
import argparse
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--layout",
        help="The wind farm layout file (path or static)",
        default="test_farm_67.csv",
    )
    parser.add_argument(
        "-s",
        "--states",
        help="The timeseries input file (path or static)",
        default="timeseries_8000.csv.gz",
    )
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes models", default=None, nargs="+"
    )
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["Jensen_linear_k007"],
        nargs="+",
    )
    parser.add_argument("-f", "--frame", help="The wake frame", default="rotor_wd")
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=[], nargs="+"
    )
    parser.add_argument(
        "-sl",
        "--show_layout",
        help="Flag for showing layout figure",
        action="store_true",
    )
    parser.add_argument(
        "-ts",
        "--timestep",
        help="The timestep of the input timeseries or data in minutes",
        default=60 * 24 * 365,  # default is one year
    )
    parser.add_argument(
        "-it", "--iterative", help="Use iterative algorithm", action="store_true"
    )
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
        default=5000,
        type=int,
    )
    parser.add_argument(
        "-nf", "--nofig", help="Do not show figures", action="store_true"
    )
    args = parser.parse_args()

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    states = foxes.input.states.Timeseries(
        data_source=args.states,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
        fixed_vars={FV.RHO: 1.225},
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_from_file(
        farm, args.layout, turbine_models=args.tmodels + [ttype.name]
    )

    if not args.nofig and args.show_layout:
        ax = foxes.output.FarmLayoutOutput(farm).get_figure()
        plt.show()
        plt.close(ax.get_figure())

    Algo = foxes.algorithms.Iterative if args.iterative else foxes.algorithms.Downwind
    algo = Algo(
        farm,
        states=states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame=args.frame,
        partial_wakes=args.pwakes,
        mbook=mbook,
    )

    cluster_pars = {
        "nodes": 1,  # number of nodes
        "cores": 24,  # number of cores per node
        "processes": 4,  # number of workers per node
        "memory": "64GB",  # memory per node
        "walltime": "00:00:10",
        "queue": "cfds.p",
        # "silence_logs": "info", # print all info log
    }

    with foxes.Engine.new(
        engine_type="slurm_cluster",
        n_procs=4,
        cluster_pars=cluster_pars,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
    ):
        time0 = time.time()
        farm_results = algo.calc_farm()
        time1 = time.time()

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
