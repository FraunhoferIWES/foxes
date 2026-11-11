import numpy as np
import argparse
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV

if __name__ == "__main__":
    # define arguments and options:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws", help="The wind speed", type=float, default=9.0)
    parser.add_argument("--wd", help="The wind direction", type=float, default=270.0)
    parser.add_argument("--ti", help="The TI value", type=float, default=0.08)
    parser.add_argument("--rho", help="The air density", type=float, default=1.225)
    parser.add_argument(
        "-dx", help="The turbine spacing in x", type=float, default=800.0
    )
    parser.add_argument(
        "-dy", help="The turbine spacing in y", type=float, default=400.0
    )
    parser.add_argument(
        "-nx", help="The number of turbines in x direction", type=int, default=6
    )
    parser.add_argument(
        "-ny", help="The number of turbines in y direction", type=int, default=10
    )
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=[], nargs="+"
    )
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["RHB", "Bastankhah2014_linear_lim_k004"],
        nargs="+",
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes models", default=None, nargs="+"
    )
    parser.add_argument("-f", "--frame", help="The wake frame", default="rotor_wd")
    parser.add_argument("-v", "--var", help="The plot variable", default=FV.WS)
    parser.add_argument(
        "-nit",
        "--not_iterative",
        help="Don't use the iterative algorithm",
        action="store_true",
    )
    parser.add_argument(
        "-nf", "--nofig", help="Do not show figures", action="store_true"
    )

    # Select chunk outputs by their model-book names.
    parser.add_argument(
        "-o", "--outputs", help="The output models", nargs="+", default=["farm_results"]
    )
    args = parser.parse_args()

    # create model book
    mbook = foxes.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    # create states
    states = foxes.input.states.SingleStateStates(
        ws=args.ws, wd=args.wd, ti=args.ti, rho=args.rho
    )

    # create wind farm
    print("\nCreating wind farm")
    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_grid(
        farm=farm,
        xy_base=[0.0, 0.0],
        step_vectors=[[args.dx, 0.0], [0.0, args.dy]],
        steps=[args.nx, args.ny],
        turbine_models=["DTU10MW"],
        verbosity=0,
    )

    # create algorithm
    Algo = (
        foxes.algorithms.Downwind if args.not_iterative else foxes.algorithms.Iterative
    )
    algo = Algo(
        farm,
        states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame=args.frame,
        partial_wakes=args.pwakes,
        mbook=mbook,
        verbosity=1,
    )

    # calculate farm results
    farm_results = algo.calc_farm(outputs=args.outputs)

    # Evaluate collected farm data only; writer-only outputs are already complete.
    if isinstance(farm_results, dict):
        farm_results = farm_results.get("farm_results")
    if farm_results is None:
        raise SystemExit(0)

    print("\nResults data:\n", farm_results)

    # add capacity and efficiency to farm results

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
                FV.CT,
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
    print(f"Annual farm yield : {turbine_results[FV.YLD].sum():.2f} GWh.")

    # horizontal flow plot
    if not args.nofig:
        xmin = -2000
        xmax = (args.nx - 1) * args.dx + 2000

        g = algo.calc_outputs(
            outputs=foxes.models.outputs.FlowPlots2D(
                plane="xy",
                reduction="states",
                plot=True,
                data_format="numpy",
                var=args.var,
                grid_pars=dict(resolution=20, xmin=xmin, xmax=xmax, yspace=1000.0),
                plot_pars=dict(figsize=(7, 5), levels=40, rotor_color="black"),
            ),
            farm_results=farm_results,
        )
        fig = g[0]
        plt.show()
        plt.close(fig)

        # center line plot:
        H = mbook.turbine_types[ttype.name].H
        n_points = 10000
        points = np.zeros((1, n_points, 3))
        points[:, :, 0] = np.linspace(xmin, xmax, n_points)[None, :]
        points[:, :, 2] = H
        point_results = algo.calc_points(farm_results, points)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(points[0, :, 0], point_results[FV.WS][0, :])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("Wind speed [m/s]")
        plt.show()
        plt.close(fig)

        # front line plot:
        points = np.zeros((1, n_points, 3))
        points[:, :, 0] = -200
        points[:, :, 1] = np.linspace(-500.0, args.ny * args.dy + 500, n_points)[
            None, :
        ]
        points[:, :, 2] = H
        point_results = algo.calc_points(farm_results, points)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(point_results[FV.WS][0, :], points[0, :, 1])
        ax.set_ylabel("y [m]")
        ax.set_xlabel("Wind speed [m/s]")
        plt.show()
        plt.close(fig)
