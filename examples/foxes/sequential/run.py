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
        "-a", "--animation", help="Write flow animation file", action="store_true"
    )
    parser.add_argument(
        "-l",
        "--layout",
        help="The wind farm layout file (path or static)",
        default="test_farm_67.csv",
    )
    parser.add_argument(
        "-s",
        "--states",
        help="The timeseries input file (path or static)",
        default="timeseries_3000.csv.gz",
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
        "-w",
        "--wakes",
        help="The wake models",
        default=["Jensen_linear_k007"],
        nargs="+",
    )
    parser.add_argument("-f", "--frame", help="The wake frame", default="rotor_wd")
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=[], nargs="+"
    )
    parser.add_argument(
        "-sl",
        "--show_layout",
        help="Flag for showing layout figure",
        action="store_true",
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

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    N = 100
    states = foxes.input.states.Timeseries(
        data_source=args.states,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "WS", FV.WD: "WD", FV.TI: "TI", FV.RHO: "RHO"},
        states_sel=range(100),
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_from_file(
        farm, args.layout, turbine_models=args.tmodels + [ttype.name]
    )

    if args.show_layout:
        ax = foxes.output.FarmLayoutOutput(farm).get_figure()
        plt.show()
        plt.close(ax.get_figure())

    algo = foxes.algorithms.Sequential(
        mbook,
        farm,
        states=states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame=args.frame,
        partial_wakes_model=args.pwakes,
        chunks={FC.STATE: None, FC.POINT: args.chunksize_points}
    )

    points = np.random.uniform(0, 1000, (N,5,3))

    with DaskRunner(
        scheduler=args.scheduler,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
    ) as runner:
        

        
        aiter = algo.iter(points=points)
        for r in aiter:
            print(aiter.index)
        
        print("\nFarm results:\n")
        print(aiter.farm_results)

        print("\nPoint results:\n")
        print(aiter.point_results)

        

        if args.animation:
            print("\nCalculating animation")

            fig, axs = plt.subplots(
                2, 1, figsize=(5.2, 7), gridspec_kw={"height_ratios": [3, 1]}
            )

            anim = foxes.output.Animator(fig)
            of = foxes.output.FlowPlots2D(algo, farm_results, runner=runner)
            anim.add_generator(
                of.gen_states_fig_xy(
                    FV.WS,
                    resolution=30,
                    quiver_pars=dict(angles="xy", scale_units="xy", scale=0.013),
                    quiver_n=35,
                    xmax=5000,
                    ymax=5000,
                    fig=fig,
                    ax=axs[0],
                    ret_im=True,
                    title=None,
                    animated=True,
                )
            )
            anim.add_generator(
                o.gen_stdata(
                    turbines=[4, 7],
                    variable=FV.REWS,
                    fig=fig,
                    ax=axs[1],
                    ret_im=True,
                    legloc="upper left",
                    animated=True,
                )
            )

            ani = anim.animate()

            lo = foxes.output.FarmLayoutOutput(farm)
            lo.get_figure(
                fig=fig,
                ax=axs[0],
                title="",
                annotate=1,
                anno_delx=-120,
                anno_dely=-60,
                alpha=0,
            )

            fpath = "ani.gif"
            print("Writing file", fpath)
            ani.save(filename=fpath, writer="pillow")