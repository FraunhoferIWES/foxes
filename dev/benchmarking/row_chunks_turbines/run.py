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
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-w", "--wake", help="The wind wake model", default="Bastankhah_linear"
    )
    parser.add_argument("-l", "--label", help="The label in table", default=None)
    parser.add_argument(
        "-c", "--chunksize", help="The maximal chunk size", type=int, default=500
    )
    parser.add_argument(
        "-o", "--ofile", help="The output file name", default="results.csv"
    )
    parser.add_argument(
        "-f", "--force", help="Overwrite output file", action="store_true"
    )
    parser.add_argument(
        "--n_cpus", help="The number of processors", type=int, default=None
    )
    parser.add_argument(
        "--threads_per_cpu",
        help="The number of threads per cpu",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    n_s = args.n_s
    n_t = args.n_t
    s = args.scheduler
    l = args.label if args.label is not None else s
    c = args.chunksize
    p0 = np.array([0.0, 0.0])
    stp = np.array([500.0, 0.0])
    ofile = Path(args.ofile)
    rotor = args.rotor
    wakes = [args.wake]

    if ofile.is_file() and not args.force:
        tresults = pd.read_csv(ofile).set_index(
            ["scheduler", "n_turbines", "chunksize"]
        )
    else:
        minds = pd.MultiIndex.from_product(
            [[l], [n_t], [c]], names=["scheduler", "n_turbines", "chunksize"]
        )
        tresults = pd.DataFrame(
            index=minds, columns=["n_states", "n_cpus", "n_threads", "time"]
        )

    idx = pd.IndexSlice
    tresults.loc[idx[l, n_t, c], "n_states"] = n_s
    tresults.loc[idx[l, n_t, c], "n_cpus"] = args.n_cpus
    tresults.loc[idx[l, n_t, c], "n_threads"] = args.threads_per_cpu

    if s == "distributed":
        cluster = LocalCluster(
            n_workers=args.n_cpus, threads_per_worker=args.threads_per_cpu
        )
        client = Client(cluster)

    print(f"\nRUNNING SCAN TASK {s} {n_t}")

    with dask.config.set(scheduler=s):

        if s == "distributed":
            print(cluster)

        ck = {FV.STATE: c}

        mbook = foxes.models.ModelBook()
        mbook.turbine_types["TOYT"] = foxes.models.turbine_types.PCtFile(
            name="TOYT", filepath="toyTurbine.csv", D=120.0, H=100.0
        )

        states = foxes.input.states.ScanWS(
            ws_list=np.linspace(3.0, 30.0, n_s), wd=270.0, ti=0.08, rho=1.225
        )

        farm = foxes.WindFarm()
        foxes.input.farm_layout.add_row(
            farm=farm,
            xy_base=p0,
            xy_step=stp,
            n_turbines=n_t,
            turbine_models=["kTI_02", "TOYT"],
            verbosity=0,
        )

        algo = foxes.algorithms.Downwind(
            mbook,
            farm,
            states=states,
            rotor_model=rotor,
            turbine_order="order_wd",
            wake_models=wakes,
            wake_frame="rotor_wd",
            partial_wakes_model="rotor_points",
            chunks=ck,
            verbosity=0,
        )

        time0 = time.time()

        with ProgressBar():
            data = algo.calc_farm()

        time1 = time.time()
        print("\nCalc time =", time1 - time0, "\n")
        tresults.loc[idx[l, n_t, c], "time"] = time1 - time0

        print(tresults)

        if s == "distributed":
            client.shutdown()

    tresults["n_states"] = tresults["n_states"].astype(int)
    # tresults['n_cpus'] = tresults['n_cpus'].astype(int)

    print()
    print("TRESULTS\n")
    print(tresults)

    print("\nWriting file", ofile)
    tresults.to_csv(ofile)
