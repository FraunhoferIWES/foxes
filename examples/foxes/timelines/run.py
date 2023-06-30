import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import foxes
import foxes.variables as FV
import foxes.constants as FC
from foxes.utils.runners import DaskRunner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nt", "--n_turbines", help="The number of turbines", default=9, type=int
    )
    parser.add_argument(
        "-s",
        "--states",
        help="The timeseries input file (path or static)",
        default="timeseries_1000.csv",
    )
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
        "-c", "--chunksize", help="The maximal chunk size", type=int, default=300
    )
    parser.add_argument(
        "-cp",
        "--chunksize_points",
        help="The maximal chunk size for points",
        type=int,
        default=5000,
    )
    parser.add_argument("-sc", "--scheduler", help="The scheduler choice", default=None)
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["Bastankhah_linear_k004"],
        nargs="+",
    )
    parser.add_argument("-f", "--frame", help="The wake frame", default="timelines")
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=[], nargs="+"
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
        "-sl",
        "--show_layout",
        help="Flag for showing layout figure",
        action="store_true",
    )
    parser.add_argument(
        "--nodask", help="Use numpy arrays instead of dask arrays", action="store_true"
    )
    parser.add_argument(
        "-ts",
        "--timestep",
        help="The timestep of the input timeseries or data in minutes",
        default=60 * 24 * 365,  # default is one year
    )
    args = parser.parse_args()

    cks = (
        None
        if args.nodask
        else {FC.STATE: args.chunksize, FC.POINT: args.chunksize_points}
    )

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    states = foxes.input.states.Timeseries(
        data_source=args.states,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
        fixed_vars={FV.RHO: 1.225, FV.TI: 0.07},
        #states_sel=range(20,30)
    )

    farm = foxes.WindFarm()
    N = int(args.n_turbines**0.5)
    foxes.input.farm_layout.add_grid(
        farm,
        xy_base=np.array([0.0, 0.0]),
        step_vectors=np.array([[3000.0, 0], [0, 4000.0]]),
        steps=(N, N),
        turbine_models=args.tmodels + [ttype.name],
    )

    if args.show_layout:
        ax = foxes.output.FarmLayoutOutput(farm).get_figure()
        plt.show()
        plt.close(ax.get_figure())

    algo = foxes.algorithms.Iterative2(
        mbook,
        farm,
        states=states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame=args.frame,
        partial_wakes_model=args.pwakes,
        chunks=cks,
    )

    with DaskRunner(
        scheduler=args.scheduler,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
    ) as runner:
        
        time0 = time.time()
        farm_results = runner.run(algo.calc_farm)
        time1 = time.time()

        print("\nCalc time =", time1 - time0, "\n")

        o = foxes.output.FarmResultsEval(farm_results)
        o.add_capacity(algo)
        o.add_capacity(algo, ambient=True)
        o.add_efficiency()

        print("\nFarm results:\n")
        print(farm_results)

        # state-turbine results
        farm_df = farm_results.to_dataframe()
        print("\nFarm results data:\n")
        print(
            farm_df[
                [
                    FV.X,
                    FV.Y,
                    FV.WD,
                    FV.AMB_REWS,
                    FV.REWS,
                    FV.AMB_TI,
                    FV.TI,
                    FV.AMB_P,
                    FV.P,
                    FV.EFF,
                ]
            ]
        )
        print()

        fig, ax = plt.subplots(figsize=(8, 8))
        o = foxes.output.FlowPlots2D(algo, farm_results)
        ims = []
        for fig, im in o.gen_states_fig_xy(
            FV.WS,
            resolution=50,
            quiver_pars=dict(angles="xy", scale_units="xy", scale=0.013),
            quiver_n=35,
            xspace=1000,
            yspace=1000,
            fig=fig,
            ax=ax,
            ret_im=True,
            animated=True,
        ):
            ims.append(im)

        ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
                                    repeat_delay=2000)
    
        fpath = "ani.gif"
        print("Writing file", fpath)
        ani.save(filename=fpath, writer="pillow")

