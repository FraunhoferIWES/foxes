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
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=[], nargs="+"
    )
    parser.add_argument(
        "-sl",
        "--show_layout",
        help="Flag for showing layout figure",
        action="store_true",
    )
    parser.add_argument("-e", "--engine", help="The engine", default="process")
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
        "-ts",
        "--timestep",
        help="The timestep of the input timeseries or data in minutes",
        default=60 * 24 * 365,  # default is one year
    )
    parser.add_argument(
        "-it", "--iterative", help="Use iterative algorithm", action="store_true"
    )
    parser.add_argument(
        "-nf", "--nofig", help="Do not show figures", action="store_true"
    )

    # Select chunk outputs by their model-book names.
    parser.add_argument(
        "-o",
        "--outputs",
        help="The outputs",
        nargs="+",
        default=["farm_results"],
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
        farm,
        args.layout,
        col_wind_farm="subfarm",
        col_cluster="cluster",
        turbine_models=args.tmodels + [ttype.name],
    )

    if args.show_layout:
        ax = foxes.algorithms.Downwind(
            farm,
            foxes.input.states.SingleStateStates(ws=8, wd=270, ti=0.08, rho=1.225),
            wake_models=[],
            mbook=mbook,
            verbosity=0,
        ).calc_outputs(
            outputs=foxes.models.outputs.FarmLayoutOutput(
                operation="get_figure", parameters=dict()
            )
        )
        plt.show()
        plt.close(ax.get_figure())

    Algo = foxes.algorithms.Iterative if args.iterative else foxes.algorithms.Downwind
    algo_kwargs = {"max_it": None} if args.iterative else {}
    algo = Algo(
        farm,
        states=states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame=args.frame,
        partial_wakes=args.pwakes,
        mbook=mbook,
        verbosity=1,
        **algo_kwargs,
    )

    engine = foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
        verbosity=2,
    )

    with engine:
        time0 = time.time()
        farm_results = algo.calc_farm(outputs=args.outputs)
        time1 = time.time()

    # Evaluate collected farm data only; writer-only outputs are already complete.
    if isinstance(farm_results, dict):
        farm_results = farm_results.get("farm_results")
    if farm_results is None:
        raise SystemExit(0)

    print("\nFarm results:\n")
    print(farm_results)

    print("\nCalc time =", time1 - time0, "\n")

    quit()

    farm_results = algo.calc_outputs(
        outputs=foxes.models.outputs.FarmResultsEval(
            operation="add_capacity_factor", parameters=dict(ambient=False)
        ),
        farm_results=farm_results,
    )
    farm_results = algo.calc_outputs(
        outputs=foxes.models.outputs.FarmResultsEval(
            operation="add_capacity_factor", parameters=dict(ambient=True)
        ),
        farm_results=farm_results,
    )
    farm_results = algo.calc_outputs(
        outputs=foxes.models.outputs.FarmResultsEval(
            operation="add_efficiency", parameters=dict()
        ),
        farm_results=farm_results,
    )

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
    turbine_results = algo.calc_outputs(
        outputs=foxes.models.outputs.FarmResultsEval(
            operation="reduce_states",
            parameters=dict(
                vars_op={
                    FV.AMB_P: "weights",
                    FV.P: "weights",
                    FV.AMB_CAPF: "weights",
                    FV.CAPF: "weights",
                }
            ),
        ),
        farm_results=farm_results,
    )
    turbine_results[FV.AMB_YLD] = algo.calc_outputs(
        outputs=foxes.models.outputs.FarmResultsEval(
            operation="calc_yield", parameters=dict(annual=True, ambient=True)
        ),
        farm_results=farm_results,
    )
    turbine_results[FV.YLD] = algo.calc_outputs(
        outputs=foxes.models.outputs.FarmResultsEval(
            operation="calc_yield", parameters=dict(annual=True)
        ),
        farm_results=farm_results,
    )
    turbine_results[FV.EFF] = turbine_results[FV.P] / turbine_results[FV.AMB_P]
    print("\nResults by turbine:\n")
    print(turbine_results)

    # power results
    P0 = algo.calc_outputs(
        outputs=foxes.models.outputs.FarmResultsEval(
            operation="calc_mean_farm_power", parameters=dict(ambient=True)
        ),
        farm_results=farm_results,
    )
    P = algo.calc_outputs(
        outputs=foxes.models.outputs.FarmResultsEval(
            operation="calc_mean_farm_power", parameters=dict()
        ),
        farm_results=farm_results,
    )
    print(f"\nFarm power        : {P / 1000:.1f} MW")
    print(f"Farm ambient power: {P0 / 1000:.1f} MW")
    print(
        f"Farm efficiency   : {algo.calc_outputs(outputs=foxes.models.outputs.FarmResultsEval(operation='calc_farm_efficiency', parameters=dict()), farm_results=farm_results) * 100:.2f} %"
    )
    print(f"Annual farm yield : {turbine_results[FV.YLD].sum():.2f} GWh")

    if not args.nofig:
        o = foxes.output.StateTurbineMap(farm_results)
        ax = o.plot_map(cmap="inferno", figsize=(6, 7))
        plt.show()

    o = foxes.output.WindFarmsEval(farm, farm_results, algo=algo)
    print("\nWind farms evaluation:\n")
    print(o.results)

    o = foxes.output.ClusterEval(farm, farm_results, algo=algo)
    print("\nCluster evaluation:\n")
    print(o.results)
