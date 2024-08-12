import time
import argparse
import matplotlib.pyplot as plt
import numpy as np

import foxes
import foxes.variables as FV
import foxes.constants as FC
from foxes.utils.runners import DaskRunner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file_pattern",
        help="The search pattern for input *.nc files",
        default="wind_rotation.nc",
    )
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes models", default="centre", nargs="+"
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
        "-wf", "--wake_frame", help="The wake frame choice", default="streamlines_100"
    )
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=[], nargs="+"
    )
    parser.add_argument(
        "-nt", "--n_turbines", help="The number of turbines", default=9, type=int
    )
    parser.add_argument(
        "-npl", "--no_pre_load", help="Pre-load the nc data", action="store_true"
    )
    parser.add_argument(
        "-e", "--engine", help="The engine", default=None
    )
    parser.add_argument(
        "-n", "--n_cpus", help="The number of cpus", default=None, type=int
    )
    parser.add_argument(
        "-c", "--chunksize_states", help="The chunk size for states", default=None, type=int
    )
    parser.add_argument(
        "-C", "--chunksize_points", help="The chunk size for points", default=None, type=int
    )
    parser.add_argument(
        "-nf", "--nofig", help="Do not show figures", action="store_true"
    )
    args = parser.parse_args()

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    states = foxes.input.states.FieldDataNC(
        args.file_pattern,
        states_coord="state",
        x_coord="x",
        y_coord="y",
        h_coord="h",
        time_format=None,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2ncvar={FV.WS: "ws", FV.WD: "wd"},
        fixed_vars={FV.RHO: 1.225, FV.TI: 0.1},
        pre_load=not args.no_pre_load,
        bounds_error=False,
    )

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
        farm,
        states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame=args.wake_frame,
        partial_wakes=args.pwakes,
        mbook=mbook,
        engine=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
    )


    time0 = time.time()
    farm_results = algo.calc_farm()
    time1 = time.time()

    print("\nCalc time =", time1 - time0, "\n")

    print(farm_results, "\n")

    fr = farm_results.to_dataframe()
    print(fr[[FV.X, FV.Y, FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]])

    if not args.nofig:
        o = foxes.output.FlowPlots2D(algo, farm_results)
        for fig in o.gen_states_fig_xy(
            FV.WS,
            resolution=10,
            figsize=(8, 8),
            quiver_pars=dict(angles="xy", scale_units="xy", scale=0.07),
            quiver_n=15,
            xmin=0,
            xmax=2500,
            ymin=0,
            ymax=2500,
        ):
            plt.show()
            plt.close(fig)
