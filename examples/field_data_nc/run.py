
import time
import argparse
import dask
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
import numpy as np

import foxes
import foxes.variables as FV
from dask.distributed import Client, LocalCluster

def run_foxes(args):

    cks = None if args.nodask else {FV.STATE: args.chunksize, "point": args.chunksize_points}

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    states = foxes.input.states.FieldDataNC(
        file_pattern=args.file_pattern,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        #var2ncvar={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
        fixed_vars={FV.RHO: 1.225},
        pre_load=not args.no_pre_load
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_grid(
        farm,
        xy_base = np.array([0., 0.]),
        step_vectors = np.array([[500., 0], [0, 500.]]),
        steps = (args.n_turbines - 1, args.n_turbines - 1),
        turbine_models=args.tmodels + [ttype.name]
    )
        
    algo = foxes.algorithms.Downwind(
                mbook,
                farm,
                states=states,
                rotor_model=args.rotor,
                turbine_order="order_wd",
                wake_models=args.wakes,
                wake_frame="rotor_wd",
                partial_wakes_model=args.pwakes,
                chunks=cks
            )
    
    time0 = time.time()

    with ProgressBar():
        farm_results = algo.calc_farm(vars_to_amb=[FV.REWS, FV.P])

    time1 = time.time()
    print("\nCalc time =",time1 - time0, "\n")

    print(farm_results)

    fr = farm_results.to_dataframe()
    print(fr[[FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]])

    o   = foxes.output.FlowPlots2D(algo, farm_results)
    fig = o.get_mean_fig_horizontal(FV.WS, resolution=10)
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file_pattern", help="The search pattern for input *.nc files")
    parser.add_argument("-t", "--turbine_file", help="The P-ct-curve csv file (path or static)", default="NREL-5MW-D126-H90.csv")
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument("-p", "--pwakes", help="The partial wakes model", default="rotor_points")
    parser.add_argument("-c", "--chunksize", help="The maximal chunk size", type=int, default=1000)
    parser.add_argument("-cp", "--chunksize_points", help="The maximal chunk size for points", type=int, default=4000)
    parser.add_argument("-sc", "--scheduler", help="The scheduler choice", default=None)
    parser.add_argument("-w", "--wakes", help="The wake models", default=['Jensen_linear_k007'], nargs='+')
    parser.add_argument("-m", "--tmodels", help="The turbine models", default=[], nargs='+')
    parser.add_argument("-nt", "--n_turbines", help="The number of turbines", default=4, type=int)
    parser.add_argument("-npl", "--no_pre_load", help="Pre-load the nc data", action="store_true")
    parser.add_argument("-n", "--n_workers", help="The number of workers for distributed run", type=int, default=None)
    parser.add_argument("-tw", "--threads_per_worker", help="The number of threads per worker for distributed run", type=int, default=None)
    parser.add_argument("--nodask", help="Use numpy arrays instead of dask arrays", action="store_true")
    args  = parser.parse_args()
    
    # parallel run:
    if args.scheduler == 'distributed' or args.n_workers is not None:
        
        print("Launching dask cluster..")
        with LocalCluster(
                n_workers=args.n_workers, 
                processes=True,
                threads_per_worker=args.threads_per_worker
            ) as cluster, Client(cluster) as client:

            print(cluster)
            print(f"Dashboard: {client.dashboard_link}\n")
            run_foxes(args)
            print("\n\nShutting down dask cluster")

    # serial run:
    else:
        with dask.config.set(scheduler=args.scheduler):
            run_foxes(args)
