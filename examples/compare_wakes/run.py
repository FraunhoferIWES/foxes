import numpy as np
import argparse
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV


def calc(mbook, farm, states, wakes, points, args):

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        wake_models=wakes,
        rotor_model=args.rotor,
        wake_frame="rotor_wd",
        partial_wakes=args.pwakes,
        mbook=mbook,
        verbosity=0,
    )

    farm_results = algo.calc_farm()
    point_results = algo.calc_points(farm_results, points[None, :])

    mbook.finalize(algo, verbosity=0)

    return point_results[args.var].to_numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dists_D",
        help="turbine distances in units of D",
        type=float,
        nargs="+",
        default=[2.0, 5.0, 8.0, 12.0, 15.0, 20.0],
    )
    parser.add_argument(
        "-y",
        "--span_y",
        help="Span in y direction in units of D",
        type=float,
        default=4,
    )
    parser.add_argument(
        "-sp", "--step", help="Point step size in units of D", type=float, default=0.01
    )
    parser.add_argument(
        "-pc", "--plot_cols", help="Columns in the plot", type=int, default=3
    )
    parser.add_argument(
        "-hg", "--height", help="The point height", type=float, default=None
    )
    parser.add_argument("-v", "--var", help="The plot variable", default=FV.WS)
    parser.add_argument(
        "-fs", "--figsize_scale", help="Scale for single D plot", type=int, default=4
    )
    parser.add_argument("--ws", help="The wind speed", type=float, default=9.0)
    parser.add_argument("--wd", help="The wind direction", type=float, default=270.0)
    parser.add_argument("--ti", help="The TI value", type=float, default=0.05)
    parser.add_argument("--rho", help="The air density", type=float, default=1.225)
    parser.add_argument("--ct", help="Set CT by hand", default=None, type=float)
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=[
            "Bastankhah2014B_linear_k004",
            "Bastankhah2014_linear_k004",
            "Bastankhah2016_linear_k004",
        ],
        nargs="+",
    )
    parser.add_argument(
        "-ew",
        "--ewakes",
        help="The extra wake models",
        default=[],
        nargs="+",
    )
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=[], nargs="+"
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes models", default="centre", nargs="+"
    )
    parser.add_argument(
        "-dfz",
        "--deficit",
        help="Plot the wind deficit instead of wind speed",
        action="store_true",
    )
    parser.add_argument("-e", "--engine", help="The engine", default="ProcessEngine")
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
    args = parser.parse_args()

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype
    D = ttype.D
    H = ttype.H

    models = args.tmodels + [ttype.name]
    if args.ct is not None:
        mbook.turbine_models["set_ct"] = foxes.models.turbine_models.SetFarmVars()
        mbook.turbine_models["set_ct"].add_var(FV.CT, args.ct)
        models.append("set_ct")

    states = foxes.input.states.SingleStateStates(
        ws=args.ws, wd=args.wd, ti=args.ti, rho=args.rho
    )

    farm = foxes.WindFarm()
    farm.add_turbine(
        foxes.Turbine(
            xy=np.array([0.0, 0.0]),
            turbine_models=models,
        )
    )

    #  y lines:

    print("\nCalculating y lines\n")

    xlist = np.array(args.dists_D) * D
    ylist = np.arange(-args.span_y, args.span_y + args.step, args.step) * D
    nd = len(args.dists_D)
    nx = len(xlist)
    ny = len(ylist)
    points = np.zeros((nd, ny, 3))
    points[:, :, 2] = args.height if args.height is not None else H
    points[:, :, 0] = xlist[:, None]
    points[:, :, 1] = ylist[None, :]
    points = points.reshape(nd * ny, 3)

    ncols = min(args.plot_cols, nd)
    nrows = int(nd / ncols)
    while nrows * ncols < nd:
        nrows += 1

    figsize = (ncols * args.figsize_scale, nrows * args.figsize_scale)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    with foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
    ):
        for wake in args.wakes:
            wakes = [wake] + args.ewakes
            print("Calculating:", wakes)

            results = calc(mbook, farm, states, wakes, points, args).reshape(nd, ny)

            for di, d in enumerate(args.dists_D):
                if nrows == 1 or ncols == 1:
                    ax = axs[di]
                else:
                    xi = int(di / ncols)
                    yi = di % ncols
                    ax = axs[xi, yi]

                if args.deficit:
                    dfz = (args.ws - results[di]) / args.ws
                    ax.plot(ylist / D, dfz, label=wake)
                    ax.set_ylabel("WS deficit")
                else:
                    ax.plot(ylist / D, results[di], label=wake)
                    ax.set_ylabel(args.var)

                ax.set_title(f"x = {d} D")
                ax.set_xlabel("y/D")
                ax.grid()

        ax.legend(loc="best")
        if not args.nofig:
            plt.show()
        plt.close(fig)

        # x line:

        print("\nCalculating x line\n")

        xlist = np.arange(-1, args.dists_D[-1] + args.step, args.step) * D
        nx = len(xlist)
        points = np.zeros((nx, 3))
        points[:, 2] = args.height if args.height is not None else H
        points[:, 0] = xlist

        figsize = (args.plot_cols * args.figsize_scale, args.figsize_scale)
        fig, ax = plt.subplots(figsize=figsize)

        for wake in args.wakes:
            wakes = [wake] + args.ewakes
            print("Calculating:", wakes)

            results = calc(mbook, farm, states, wakes, points, args)

            if args.deficit:
                dfz = (args.ws - results[0]) / args.ws
                ax.plot(xlist / D, dfz, label=wake)
                ax.set_ylabel("WS deficit")
            else:
                ax.plot(xlist / D, results[0], label=wake)
                ax.set_ylabel(args.var)

            ax.set_title(f"y = 0")
            ax.set_xlabel("x/D")
            ax.legend(loc="best")
            ax.grid()

    if not args.nofig:
        plt.show()
