import numpy as np
import argparse
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV


def main(argv=None):
    """Run this tutorial with an explicit argument list or command-line options."""
    # define arguments and options:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", default=None)
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
    parser.add_argument("--var", help="The plot variable", default=FV.WS)
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
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=2,
        help="0 quiet, 1 summary, 2 reports, 3 library progress",
    )
    args = parser.parse_args(argv)
    if args.verbosity > 0:
        print("Running example: induction")

    # create model book
    mbook = foxes.ModelBook()
    mbook.outputs["farm_results"].fdata_variables.extend([FV.AMB_WD, FV.EFF])

    if args.nofig:
        for name in args.outputs:
            output = mbook.outputs[name]
            if hasattr(output, "show"):
                output.show = False
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    # create states
    states = foxes.input.states.SingleStateStates(
        ws=args.ws, wd=args.wd, ti=args.ti, rho=args.rho
    )

    # create wind farm
    if args.verbosity > 1:
        print("\nCreating wind farm")
    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_grid(
        farm=farm,
        xy_base=[0.0, 0.0],
        step_vectors=[[args.dx, 0.0], [0.0, args.dy]],
        steps=[args.nx, args.ny],
        turbine_models=["DTU10MW"],
        verbosity=max(args.verbosity - 1, 0),
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
        verbosity=max(args.verbosity - 1, 0),
    )

    engine = foxes.Engine.new(args.engine, verbosity=max(args.verbosity - 1, 0))
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
    required = {
        FV.AMB_P,
        FV.AMB_REWS,
        FV.AMB_TI,
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
                    FV.CT,
                    FV.EFF,
                ]
            ]
        )
        print()

    # horizontal flow plot
    if not args.nofig:
        xmin = -2000
        xmax = (args.nx - 1) * args.dx + 2000

        with engine:
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
        with engine:
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
        with engine:
            point_results = algo.calc_points(farm_results, points)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(point_results[FV.WS][0, :], points[0, :, 1])
        ax.set_ylabel("y [m]")
        ax.set_xlabel("Wind speed [m/s]")
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
