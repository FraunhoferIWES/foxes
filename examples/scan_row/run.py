
import numpy as np
import time
import argparse
import dask
from dask.diagnostics import ProgressBar

import foxes
import foxes.variables as FV
from dask.distributed import Client

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("n_s", help="The number of states", type=int)
    parser.add_argument("n_t", help="The number of turbines", type=int)
    parser.add_argument("-c", "--chunksize", help="The maximal chunk size", type=int, default=1000)
    parser.add_argument("-s", "--scheduler", help="The scheduler choice", default=None)
    parser.add_argument("-w", "--n_workers", help="The number of workers for distributed run", type=int, default=None)
    parser.add_argument("-t", "--threads_per_worker", help="The number of threads per worker for distributed run", type=int, default=None)
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument("-p", "--pwakes", help="The partial wakes model", default="rotor_points")
    parser.add_argument("--nodask", help="Use numpy arrays instead of dask arrays", action="store_true")
    args  = parser.parse_args()

    n_s = args.n_s
    n_t = args.n_t
    p0  = np.array([0., 0.])
    stp = np.array([500., 0.])
    
    cks = None if args.nodask else {FV.STATE: args.chunksize}
    if args.scheduler == 'distributed':
        client = Client(n_workers=args.n_workers, threads_per_worker=args.threads_per_worker)
        print(f"\n{client}")
        print(f"Dashboard: {client.dashboard_link}\n")

    with dask.config.set(scheduler=args.scheduler):

        mbook = foxes.models.ModelBook()
        mbook.turbine_types["TOYT"] = foxes.models.turbine_types.PCtFile(
                                        name="TOYT", filepath="toyTurbine.csv", D=120., H=100.)

        states = foxes.input.states.ScanWS(
            ws_list=np.linspace(3., 30., n_s),
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
            turbine_models=["kTI_04", "TOYT"]
        )
        
        algo = foxes.algorithms.Downwind(
                    mbook,
                    farm,
                    states=states,
                    rotor_model=args.rotor,
                    turbine_order="order_wd",
                    wake_models=['Bastankhah_linear'],
                    wake_frame="mean_wd",
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
