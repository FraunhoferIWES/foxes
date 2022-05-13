
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
    parser.add_argument("lfile", help="The turbine layout file")
    parser.add_argument("tfile", help="The timeseries input file")
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument("-p", "--pwakes", help="The partial wakes model", default="rotor_points")
    parser.add_argument("-c", "--chunksize", help="The maximal chunk size", type=int, default=1000)
    parser.add_argument("-s", "--scheduler", help="The scheduler choice", default=None)
    parser.add_argument("-w", "--wakes", help="The wake models", default=['Jensen_linear_k007'], nargs='+')
    parser.add_argument("-m", "--tmodels", help="The turbine models", default=["TOYT"], nargs='+')
    parser.add_argument("-k", "--n_workers", help="The number of workers for distributed run", type=int, default=None)
    parser.add_argument("-t", "--threads_per_worker", help="The number of threads per worker for distributed run", type=int, default=None)
    parser.add_argument("--nodask", help="Use numpy arrays instead of dask arrays", action="store_true")
    args  = parser.parse_args()
    
    cks = None if args.nodask else {FV.STATE: args.chunksize}
    if args.scheduler == 'distributed':
        client = Client(n_workers=args.n_workers, threads_per_worker=args.threads_per_worker)
        print(f"\n{client}")
        print(f"Dashboard: {client.dashboard_link}\n")
    dask.config.set(scheduler=args.scheduler)

    mbook = foxes.models.ModelBook()
    mbook.turbine_types["TOYT"] = foxes.models.turbine_types.PCtFile(
                                    name="TOYT", filepath="toyTurbine.csv", D=120., H=100.)

    states = foxes.input.states.Timeseries(
        data_source=args.tfile,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
        fixed_vars={FV.RHO: 1.225}
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_from_file(
        farm,
        args.lfile,
        turbine_models=args.tmodels
    )

    #ax = foxes.output.FarmLayoutOutput(farm).get_figure()
    #plt.show()
    #plt.close(ax.get_figure())
        
    algo = foxes.algorithms.Downwind(
                mbook,
                farm,
                states=states,
                rotor_model=args.rotor,
                turbine_order="order_wd",
                wake_models=args.wakes,
                wake_frame="mean_wd",
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
