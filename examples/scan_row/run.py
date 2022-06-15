
import numpy as np
import time
import argparse
import dask
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV
from dask.distributed import Client

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("n_s", help="The number of states", type=int)
    parser.add_argument("n_t", help="The number of turbines", type=int)
    parser.add_argument("--n_p", help="The number of turbines", type=int, default=2000)
    parser.add_argument("--ws0", help="The lowest wind speed", type=float, default=3.)
    parser.add_argument("--ws1", help="The highest wind speed", type=float, default=30.)
    parser.add_argument("-t", "--turbine_file", help="The P-ct-curve csv file (path or static)", default="NREL-5MW-D126-H90.csv")
    parser.add_argument("-c", "--chunksize", help="The maximal chunk size", type=int, default=1000)
    parser.add_argument("-sc", "--scheduler", help="The scheduler choice", default=None)
    parser.add_argument("-w", "--n_workers", help="The number of workers for distributed run", type=int, default=None)
    parser.add_argument("-tw", "--threads_per_worker", help="The number of threads per worker for distributed run", type=int, default=None)
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument("-p", "--pwakes", help="The partial wakes model", default="rotor_points")
    parser.add_argument("--nodask", help="Use numpy arrays instead of dask arrays", action="store_true")
    parser.add_argument("-cl", "--calc_cline", help="Calculate centreline", action="store_true")
    parser.add_argument("-cm", "--calc_mean", help="Calculate states mean", action="store_true")
    args  = parser.parse_args()

    n_s = args.n_s
    n_t = args.n_t
    n_p = args.n_p
    p0  = np.array([0., 0.])
    stp = np.array([500., 0.])
    
    cks = None if args.nodask else {FV.STATE: args.chunksize}
    if args.scheduler == 'distributed':
        client = Client(n_workers=args.n_workers, threads_per_worker=args.threads_per_worker)
        print(f"\n{client}")
        print(f"Dashboard: {client.dashboard_link}\n")
    dask.config.set(scheduler=args.scheduler)

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype
    D = ttype.D
    H = ttype.H

    states = foxes.input.states.ScanWS(
        ws_list=np.linspace(args.ws0, args.ws1, n_s),
        wd=270.,
        ti=0.08,
        rho=1.225
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=p0, 
        xy_step=stp, 
        n_turbines=n_t,
        turbine_models=["kTI_04", ttype.name]
    )
    
    algo = foxes.algorithms.Downwind(
                mbook,
                farm,
                states=states,
                rotor_model=args.rotor,
                turbine_order="order_wd",
                wake_models=['Bastankhah_linear'],
                wake_frame="rotor_wd",
                partial_wakes_model=args.pwakes,
                chunks=cks
            )
    
    time0 = time.time()
    
    with ProgressBar():
        farm_results = algo.calc_farm()

    time1 = time.time()
    print("\nCalc time =",time1 - time0, "\n")

    print("\nFarm results:\n", farm_results)

    fr = farm_results.to_dataframe()
    print(fr[[FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]])

    if args.calc_cline:

        points          = np.zeros((n_s, n_p, 3))
        points[:, :, 0] = np.linspace(p0[0], p0[0] + n_t*stp[0] + 10*D, n_p)[None, :]
        points[:, :, 1] = p0[1]
        points[:, :, 2] = H
        print("\nPOINTS:\n", points[0])

        time0 = time.time()

        with ProgressBar():
            point_results = algo.calc_points(farm_results, points, vars_to_amb=[FV.WS, FV.TI])

        time1 = time.time()
        print("\nCalc time =",time1 - time0, "\n")

        print(point_results)

        fig, ax = plt.subplots()
        for s in range(points.shape[0]):
            ax.plot(points[s, :, 0], point_results[FV.WS][s, :])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("Wind speed [m/s]")
        ax.set_title("Centreline wind speed")
        plt.show()
        plt.close(fig)

    if args.calc_mean:
        o   = foxes.output.FlowPlots2D(algo, farm_results)
        fig = o.get_mean_fig_horizontal(FV.WS, resolution=10)
        plt.show()
