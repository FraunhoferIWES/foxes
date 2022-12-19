import numpy as np
import argparse
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV
from foxes.utils.runners import DaskRunner

def calc(farm, states, wakes, points, args):

    cks = None if args.nodask else {FV.STATE: args.chunksize}

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    algo = foxes.algorithms.Downwind(
        mbook,
        farm,
        states=states,
        rotor_model=args.rotor,
        wake_models=wakes,
        wake_frame="rotor_wd",
        partial_wakes_model=args.pwakes,
        chunks=cks,
        verbosity=0
    )

    farm_results = algo.calc_farm()
    point_results = algo.calc_points(farm_results, points[None, :])

    return point_results[args.var].to_numpy()

def run_foxes(args):

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype
    D = ttype.D
    H = ttype.H

    states = foxes.input.states.SingleStateStates(
        ws=args.ws, wd=args.wd, ti=args.ti, rho=args.rho
    )

    farm = foxes.WindFarm()
    farm.add_turbine(
        foxes.Turbine(
            xy=np.array([0.0, 0.0]), 
            turbine_models=args.tmodels + [ttype.name],
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
    points = points.reshape(nd*ny, 3) 

    ncols = min(args.plot_cols, nd)
    nrows = int(nd/ncols)
    while nrows*ncols < nd: 
        nrows += 1

    figsize = (ncols*args.figsize_scale, nrows*args.figsize_scale)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)

    for wake in args.wakes:

        wakes = [wake] + args.ewakes
        print("Calculating:", wakes)

        results = calc(farm, states, wakes, points, args).reshape(nd, ny)

        for di, d in enumerate(args.dists_D):

            if nrows == 1 or ncols == 1:
                ax = axs[di]
            else:
                xi = int(di/ncols)
                yi = di % ncols
                ax = axs[xi, yi]
            
            ax.plot(ylist/D, results[di], label=wake)

            ax.set_title(f"x = {d} D")
            ax.set_xlabel("y/D")
            ax.set_ylabel(args.var)
            ax.grid()


    ax.legend(loc="best")
    plt.show()
    plt.close(fig)

    # x line:

    print("\nCalculating x line\n")

    xlist = np.arange(-1, args.dists_D[-1] + args.step, args.step) * D
    nx = len(xlist)
    points = np.zeros((nx, 3))
    points[:, 2] = args.height if args.height is not None else H
    points[:, 0] = xlist

    figsize = (args.plot_cols*args.figsize_scale, args.figsize_scale)
    fig, ax = plt.subplots(figsize=figsize)

    for wake in args.wakes:

        wakes = [wake] + args.ewakes
        print("Calculating:", wakes)

        results = calc(farm, states, wakes, points, args)

        ax.plot(xlist/D, results[0], label=wake)

        ax.set_title(f"y = 0")
        ax.set_xlabel("x/D")
        ax.set_ylabel(args.var)
        ax.legend(loc="best")
        ax.grid()

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dists_D", help="turbine distances in units of D", type=float, nargs="+", default=[2., 6., 10.])
    parser.add_argument("-y", "--span_y", help="Span in y direction in units of D", type=float, default=4)
    parser.add_argument("-sp", "--step", help="Point step size in units of D", type=float, default=0.01)
    parser.add_argument("-pc", "--plot_cols", help="Columns in the plot", type=int, default=3)
    parser.add_argument("-hg", "--height", help="The point height", type=float, default=None)
    parser.add_argument("-v", "--var", help="The plot variable", default=FV.WS)
    parser.add_argument("-fs", "--figsize_scale", help="Scale for single D plot", type=int, default=4)
    parser.add_argument("--ws", help="The wind speed", type=float, default=9.0)
    parser.add_argument("--wd", help="The wind direction", type=float, default=270.0)
    parser.add_argument("--ti", help="The TI value", type=float, default=0.05)
    parser.add_argument("--rho", help="The air density", type=float, default=1.225)
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
        default=["Bastankhah_linear", "PorteAgel_linear"],
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
        "-m", "--tmodels", help="The turbine models", default=["kTI_04"], nargs="+"
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes model", default="auto"
    )
    parser.add_argument(
        "-c", "--chunksize", help="The maximal chunk size", type=int, default=1000
    )
    parser.add_argument("-sc", "--scheduler", help="The scheduler choice", default=None)
    parser.add_argument(
        "-n",
        "--n_workers",
        help="The number of workers for distributed run",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-tw",
        "--threads_per_worker",
        help="The number of threads per worker for distributed run",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--nodask", help="Use numpy arrays instead of dask arrays", action="store_true"
    )
    args = parser.parse_args()

    with DaskRunner(
        scheduler=args.scheduler,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
    ) as runner:

        runner.run(run_foxes, args=(args,))