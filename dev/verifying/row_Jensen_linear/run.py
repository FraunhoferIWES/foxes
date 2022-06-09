
import numpy as np
import pandas as pd
import time
import dask
import argparse
from dask.diagnostics import ProgressBar
from pathlib import Path

import foxes
import foxes.variables as FV
from dask.distributed import Client, LocalCluster

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("n_s", help="The number of states", type=int)
    parser.add_argument("n_t", help="The number of turbines", type=int)
    parser.add_argument("scheduler", help="The scheduler choice")
    parser.add_argument("-c", "--chunksize", help="The maximal chunk size", type=int, default=500)
    parser.add_argument("-C", "--cfile", help="The results file for verification", default="flappy/veri_results.csv.gz")
    parser.add_argument("--n_cpus", help="The number of processors", type=int, default=None)
    parser.add_argument("--threads_per_cpu", help="The number of threads per cpu", type=int, default=None)
    args  = parser.parse_args()

    n_s   = args.n_s
    n_t   = args.n_t
    s     = args.scheduler
    c     = args.chunksize
    p0    = np.array([0., 0.])
    stp   = np.array([500., 0.])


    if s == 'distributed':
        cluster = LocalCluster(n_workers=args.n_cpus, threads_per_worker=args.threads_per_cpu)
        client  = Client(cluster)

    print(f"\nRUNNING TASK {s} {c}")

    with dask.config.set(scheduler=s):

        if s == 'distributed':
            print(client)

        ck = {FV.STATE: c}

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
            turbine_models=["TOYT"],
            verbosity=0
        )
        
        algo = foxes.algorithms.Downwind(
                    mbook,
                    farm,
                    states=states,
                    rotor_model="centre",
                    turbine_order="order_wd",
                    wake_models=['Jensen_linear_k007'],
                    wake_frame="rotor_wd",
                    partial_wakes_model="rotor_points",
                    chunks=ck,
                    verbosity=0
                )
        
        time0 = time.time()
        
        with ProgressBar():
            data = algo.calc_farm()

        time1 = time.time()
        print("\nCalc time =",time1 - time0, "\n")

        if s == 'distributed':
            client.shutdown()

    df = data.to_dataframe()[[FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]]

    print()
    print("TRESULTS\n")
    print(df)

    print("\Reading file", args.cfile)
    fdata = pd.read_csv(args.cfile)
    print(fdata)

    print("\nVERIFYING\n")
    df[FV.WS] = df["REWS"]
    df[FV.AMB_WS] = df["AMB_REWS"]

    delta = df.reset_index() - fdata
    print(delta)
    print(delta.max())
    

