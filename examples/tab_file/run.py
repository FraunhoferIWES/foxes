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
        help="The states input file (path or static)",
        default="winds100.tab",
    )
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes models", default={}, nargs="+"
    )
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["CrespoHernandez_quadratic", "Bastankhah2014_linear"],
        nargs="+",
    )
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
        "-cm", "--calc_mean", help="Calculate states mean", action="store_true"
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

    states = foxes.input.states.TabStates(
        data_source=args.states,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.RHO: 1.225, FV.TI: 0.05},
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_from_file(
        farm,
        args.layout,
        col_x="x",
        col_y="y",
        col_H="H",
        turbine_models=[ttype.name, "kTI_02"] + args.tmodels,
    )

    engine = foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
    )

    if not args.nofig:
        with engine:
            foxes.algorithms.Downwind(
                farm, states, mbook=mbook, wake_models=[], verbosity=0
            ).calc_outputs(
                outputs=foxes.models.outputs.StatesRosePlotOutput(
                    point=[0.0, 0.0, 100.0], plot=True
                )
            )[0]
        plt.show()

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

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        mbook=mbook,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame="rotor_wd",
        partial_wakes=args.pwakes,
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

    print("\nCalc time =", time1 - time0, "\n")

    print(farm_results)

    fr = farm_results.to_dataframe()
    print(fr[[FV.WD, FV.H, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P, FV.WEIGHT]])

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
    print(
        f"Annual farm yield : {algo.calc_outputs(outputs=foxes.models.outputs.FarmResultsEval(operation='calc_farm_yield', parameters=dict()), farm_results=farm_results):.2f} GWh"
    )

    if not args.nofig and args.calc_mean:
        with engine:
            img_data = algo.calc_outputs(
                outputs=foxes.models.outputs.FlowPlots2D(
                    plane="xy",
                    reduction="mean",
                    plot=False,
                    data_format="numpy",
                    var=FV.WS,
                    grid_pars=dict(resolution=30),
                ),
                farm_results=farm_results,
            )
        fig = algo.calc_outputs(
            outputs=foxes.models.outputs.FlowPlots2D(
                plane="xy",
                reduction="mean",
                plot=True,
                data_format="numpy",
                var=FV.WS,
                grid_pars=dict(resolution=30),
                plot_pars=dict(xlabel="x [m]", ylabel="y [m]"),
            ),
            farm_results=farm_results,
        )
        plt.show()
