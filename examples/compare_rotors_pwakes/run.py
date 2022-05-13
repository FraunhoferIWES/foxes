
import numpy as np
import pandas as pd
import time
import argparse
import dask
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV
from dask.distributed import Client

def calc(args, rotor, pwake):

    farm = foxes.WindFarm()
    farm.add_turbine(foxes.Turbine(
            xy=np.array([0., 0.]),
            turbine_models=args.tmodels), verbosity=0)
    farm.add_turbine(foxes.Turbine(
            xy=np.array([args.dist_x, 0.]),
            turbine_models=["sety"] + args.tmodels,
            D=args.D,
            H=args.H), verbosity=0)

    states = foxes.input.states.StatesTable(
        sdata,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti", FV.RHO: "rho"}
    )

    mbook = foxes.models.ModelBook()
    mbook.turbine_types["TOYT"] = foxes.models.turbine_types.PCtFile(
                                    name="TOYT", filepath="toyTurbine.csv",
                                    D=args.D0, H=args.H0)

    ydata = np.full((states.size(), farm.n_turbines), np.nan)
    ydata[:, 1] = sdata["y"].to_numpy()
    mbook.turbine_models["sety"] = foxes.models.turbine_models.SetFarmVars(pre_rotor=True)
    mbook.turbine_models["sety"].add_var(FV.Y, ydata)
    
    algo = foxes.algorithms.Downwind(
                mbook,
                farm,
                states=states,
                rotor_model=rotor,
                turbine_order="order_wd",
                wake_models=args.wakes,
                wake_frame="mean_wd",
                partial_wakes_model=pwake,
                chunks=cks,
                verbosity=0
            )
    
    print(f"\nCalculating rotor = {rotor}, pwake = {pwake}")
    with ProgressBar():
        farm_results = algo.calc_farm()
    
    return farm_results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ws", help="The wind speed", type=float, default=9.0)
    parser.add_argument("--wd", help="The wind direction", type=float, default=270.0)
    parser.add_argument("--ti", help="The TI value", type=float, default=0.08)
    parser.add_argument("--rho", help="The air density", type=float, default=1.225)
    parser.add_argument("-D0", help="The rotor diameter of the source turbine", type=float, default=120.)
    parser.add_argument("-H0", help="The hub height of the source turbine", type=float, default=100.)
    parser.add_argument("-D", help="The rotor diameter of the target turbine", type=float, default=120.)
    parser.add_argument("-H", help="The hub height of the target turbine", type=float, default=100.)
    parser.add_argument("-d", "--dist_x", help="The turbine distance in x", type=float, default=500.0)
    parser.add_argument("-y0", "--ymin", help="The minimal y value", type=float, default=-1000.)
    parser.add_argument("-y1", "--ymax", help="The maximal y value", type=float, default=1000.)
    parser.add_argument("-ys", "--ystep", help="The step size in y direction", type=float, default=1.)
    parser.add_argument("-w", "--wakes", help="The wake models", default=['Bastankhah_linear_k002'], nargs='+')
    parser.add_argument("-m", "--tmodels", help="The turbine models", default=["TOYT"], nargs='+')
    parser.add_argument("-r", "--rotors", help="The rotor model(s)", default=["grid100"], nargs='+')
    parser.add_argument("-p", "--pwakes", help="The partial wakes model(s)", default=["rotor_points", "distsliced", "axiwake_5"], nargs='+')
    parser.add_argument("-c", "--chunksize", help="The maximal chunk size", type=int, default=1000)
    parser.add_argument("-s", "--scheduler", help="The scheduler choice", default=None)
    parser.add_argument("--nodask", help="Use numpy arrays instead of dask arrays", action="store_true")
    args  = parser.parse_args()

    Ny    = int( (args.ymax - args.ymin) // args.ystep )
    sdata = pd.DataFrame(index=range(Ny))
    sdata.index.name = "state"
    sdata["ws"]  = args.ws
    sdata["wd"]  = args.wd
    sdata["ti"]  = args.ti
    sdata["rho"] = args.rho
    sdata["y"]   = np.linspace(args.ymin, args.ymax, Ny)

    cks = None if args.nodask else {FV.STATE: args.chunksize}
    if args.scheduler == 'distributed':
        client = Client(n_workers=args.n_workers, threads_per_worker=args.threads_per_worker)
        print(f"\n{client}")
        print(f"Dashboard: {client.dashboard_link}\n")
    dask.config.set(scheduler=args.scheduler)

    fig, ax = plt.subplots(figsize=(10,4))
    if len(args.rotors) == 1:

        for pwake in args.pwakes:

            farm_results = calc(args, args.rotors[0], pwake)

            ax.plot(farm_results[FV.Y][:, 1], farm_results[FV.REWS][:, 1], label=pwake)

            ax.set_title(f"Rotor {args.rotors[0]}, varying partial wake models")
    
    elif len(args.pwakes) == 1:

        for rotor in args.rotors:

            farm_results = calc(args, rotor, args.pwakes[0])

            ax.plot(farm_results[FV.Y][:, 1], farm_results[FV.REWS][:, 1], label=rotor)

            ax.set_title(f"Partial wake model {args.pwakes[0]}, varying rotor models")
    
    elif len(args.rotors) == len(args.pwakes):

        for rotor, pwake in zip(args.rotors, args.pwakes):

            farm_results = calc(args, rotor, pwake)

            ax.plot(farm_results[FV.Y][:, 1], farm_results[FV.REWS][:, 1], label=f"{rotor}, {pwake}")

            ax.set_title("Varying rotor and partial wake models")

    else:
        raise ValueError(f"Please either give one rotor, or one pwake, or same number of both")

    ax.set_xlabel("y [m]")
    ax.set_ylabel("ws [m/s]")
    ax.legend()
    plt.show()
