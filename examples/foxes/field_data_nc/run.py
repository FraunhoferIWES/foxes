import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import foxes
import foxes.variables as FV
from foxes.utils.runners import DaskRunner


def run_foxes(args, states):

    cks = (
        None
        if args.nodask
        else {FV.STATE: args.chunksize, "point": args.chunksize_points}
    )

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    farm = foxes.WindFarm()
    N = int(args.n_turbines**0.5)
    foxes.input.farm_layout.add_grid(
        farm,
        xy_base=np.array([500.0, 500.0]),
        step_vectors=np.array([[500.0, 0], [0, 500.0]]),
        steps=(N, N),
        turbine_models=args.tmodels + [ttype.name],
    )

    algo = foxes.algorithms.Downwind(
        mbook,
        farm,
        states=states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame=args.wake_frame,
        partial_wakes_model=args.pwakes,
        chunks=cks,
    )

    time0 = time.time()

    farm_results = algo.calc_farm(vars_to_amb=[FV.REWS, FV.P])

    time1 = time.time()
    print("\nCalc time =", time1 - time0, "\n")

    print(farm_results)

    fr = farm_results.to_dataframe()
    print(fr[[FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]])

    o = foxes.output.FlowPlots2D(algo, farm_results)
    o.get_mean_fig_xy(FV.WS, resolution=10)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file_pattern", help="The search pattern for input *.nc files")
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes model", default="rotor_points"
    )
    parser.add_argument(
        "-c", "--chunksize", help="The maximal chunk size", type=int, default=1000
    )
    parser.add_argument(
        "-cp",
        "--chunksize_points",
        help="The maximal chunk size for points",
        type=int,
        default=4000,
    )
    parser.add_argument("-sc", "--scheduler", help="The scheduler choice", default=None)
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["Jensen_linear_k007"],
        nargs="+",
    )
    parser.add_argument(
        "-wf", "--wake_frame", help="The wake frame choice", default="rotor_wd"
    )
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=[], nargs="+"
    )
    parser.add_argument(
        "-nt", "--n_turbines", help="The number of turbines", default=9, type=int
    )
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

    """
    sdata = xr.open_mfdataset(
        args.file_pattern,
        parallel=False,
        concat_dim="Time",
        combine="nested",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )
    """
    
    states = foxes.input.states.FieldDataNC(
        args.file_pattern,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        # var2ncvar={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
        fixed_vars={FV.RHO: 1.225},
    )

    with DaskRunner(
        scheduler=args.scheduler,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
    ) as runner:

        runner.run(run_foxes, args=(args, states))
