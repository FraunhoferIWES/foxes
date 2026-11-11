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

    # Select chunk outputs by their model-book names.
    parser.add_argument(
        "-o", "--outputs", help="The output models", nargs="+", default=["farm_results"]
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

    engine = foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
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
            verbosity=0,
        ).calc_outputs(
            outputs=foxes.models.outputs.FarmLayoutOutput(
                operation="get_figure", parameters=dict(fig=fig, ax=ax1)
            )
        )

        with engine:
            foxes.algorithms.Downwind(
                farm, states, mbook=mbook, wake_models=[], verbosity=0
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
        verbosity=0,
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

    algo.calc_outputs(
        outputs=foxes.models.outputs.FarmResultsEval(
            operation="add_capacity_factor", parameters=dict(ambient=False)
        ),
        farm_results=farm_results,
    )
    algo.calc_outputs(
        outputs=foxes.models.outputs.FarmResultsEval(
            operation="add_capacity_factor", parameters=dict(ambient=True)
        ),
        farm_results=farm_results,
    )
    algo.calc_outputs(
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
