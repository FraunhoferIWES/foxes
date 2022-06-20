
import time
import argparse
import dask
from dask.diagnostics import ProgressBar

import foxes
import foxes.variables as FV
from dask.distributed import Client

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--layout", help="The wind farm layout file (path or static)", default="test_farm_67.csv")
    parser.add_argument("-s", "--states", help="The states input file (path or static)", default="states.csv.gz")
    parser.add_argument("-t", "--turbine_file", help="The P-ct-curve csv file (path or static)", default="NREL-5MW-D126-H90.csv")
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument("-p", "--pwakes", help="The partial wakes model", default="rotor_points")
    parser.add_argument("-c", "--chunksize", help="The maximal chunk size", type=int, default=1000)
    parser.add_argument("-sc", "--scheduler", help="The scheduler choice", default=None)
    parser.add_argument("-w", "--wakes", help="The wake models", default=['Jensen_linear_k007'], nargs='+')
    parser.add_argument("-m", "--tmodels", help="The turbine models", default=[], nargs='+')
    parser.add_argument("-k", "--n_workers", help="The number of workers for distributed run", type=int, default=None)
    parser.add_argument("-tw", "--threads_per_worker", help="The number of threads per worker for distributed run", type=int, default=None)
    parser.add_argument("--nodask", help="Use numpy arrays instead of dask arrays", action="store_true")
    args  = parser.parse_args()
    
    cks = None if args.nodask else {FV.STATE: args.chunksize}
    if args.scheduler == 'distributed':
        client = Client(n_workers=args.n_workers, threads_per_worker=args.threads_per_worker)
        print(f"\n{client}")
        print(f"Dashboard: {client.dashboard_link}\n")
    dask.config.set(scheduler=args.scheduler)

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    states = foxes.input.states.StatesTable(
        data_source=args.states,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO, FV.MOL],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti", FV.MOL: "mol"},
        fixed_vars={FV.RHO: 1.225, FV.Z0: 0.05, FV.H: 100.0},
        profiles={FV.WS: "ABLLogWsProfile"}
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_from_file(
        farm,
        args.layout,
        col_x="x",
        col_y="y",
        col_H="H",
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
        farm_results = algo.calc_farm()

    time1 = time.time()
    print("\nCalc time =",time1 - time0, "\n")

    print(farm_results)
    
    fr = farm_results.to_dataframe()
    print(fr[[FV.WD, FV.H, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]])

    if args.scheduler == 'distributed':
        print("\nClosing dask client")
        client.close()
        