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

    """
    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_from_file(
        farm, args.layout, turbine_models=args.tmodels + [ttype.name]
    )
    """
    farm = foxes.WindFarm()
    N = 3
    foxes.input.farm_layout.add_grid(
        farm,
        xy_base=np.array([0.0, 0.0]),
        step_vectors=np.array([[1000.0, 0], [0, 800.0]]),
        steps=(N, N),
        turbine_models=args.tmodels + [ttype.name],
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

    #points = np.random.uniform(0, 1000, (N,5,3))

    with DaskRunner(
        scheduler=args.scheduler,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
    ) as runner:
        
        # in case of animation, add a plugin that creates the images:
        plugins = []
        if args.animation:
            fig, ax = plt.subplots()
            anigen = foxes.output.SeqFlowAnimationPlugin(
                orientation="xy",
                var=FV.WS,
                resolution=10,
                quiver_pars=dict(angles="xy", scale_units="xy", scale=0.04),
                quiver_n=101,
                xmax=5000,
                ymax=5000,
                fig=fig,
                ax=ax,
                vmin=0,
                vmax=15,
                ret_im=True,
                title=None,
                animated=True,
            )
            plugins = [anigen]

        # run all states sequentially:
        aiter = algo.iter(plugins=plugins) # ,points=points)
        for r in aiter:
            print(aiter.index)
        
        print("\nFarm results:\n")
        print(aiter.farm_results)

        #print("\nPoint results:\n")
        #print(aiter.point_results)

        if args.animation:
            print("\nCalculating animation")

            anim = foxes.output.Animator(fig)
            anim.add_generator(anigen.gen_images())
            ani = anim.animate(interval=700)

            lo = foxes.output.FarmLayoutOutput(farm)
            lo.get_figure(
                fig=fig,
                ax=ax,
                title="",
                annotate=1,
                anno_delx=-120,
                anno_dely=-60,
                alpha=0,
                s=10,
            )

            fpath = "ani.gif"
            print("Writing file", fpath)
            ani.save(filename=fpath, writer="pillow")
            