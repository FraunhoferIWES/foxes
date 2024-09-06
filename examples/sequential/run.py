import argparse
import matplotlib.pyplot as plt
import numpy as np

import foxes
import foxes.variables as FV

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--animation", help="Write flow animation file", action="store_true"
    )
    parser.add_argument(
        "-d", "--debug", help="Switch on wake debugging", action="store_true"
    )
    parser.add_argument(
        "-l",
        "--layout",
        help="The wind farm layout file (path or static)",
        default="test_farm_67.csv",
    )
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes models", default=None, nargs="+"
    )
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["Bastankhah2014_linear_lim_k004"],
        nargs="+",
    )
    parser.add_argument(
        "-f", "--frame", help="The wake frame", default="seq_dyn_wakes_1min"
    )
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
        "-e", "--engine", help="The engine", default="numpy"
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

    states = foxes.input.states.Timeseries(
        data_source="timeseries_3000.csv.gz",
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "WS", FV.WD: "WD", FV.TI: "TI", FV.RHO: "RHO"},
        states_sel=range(240, 290),
    )

    farm = foxes.WindFarm()
    N = 3
    foxes.input.farm_layout.add_grid(
        farm,
        xy_base=np.array([0.0, 0.0]),
        step_vectors=np.array([[1000.0, 0], [0, 800.0]]),
        steps=(N, N),
        turbine_models=args.tmodels + [ttype.name],
    )

    if not args.nofig and args.show_layout:
        ax = foxes.output.FarmLayoutOutput(farm).get_figure()
        plt.show()
        plt.close(ax.get_figure(figsize=(8, 8)))

    algo = foxes.algorithms.Sequential(
        farm,
        states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame=args.frame,
        partial_wakes=args.pwakes,
        mbook=mbook,
        engine=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
    )

    # in case of animation, add a plugin that creates the images:
    if args.animation:
        fig, ax = plt.subplots()
        anigen = foxes.output.SeqFlowAnimationPlugin(
            orientation="xy",
            var=FV.WS,
            resolution=10,
            levels=None,
            quiver_pars=dict(scale=0.01),
            quiver_n=307,
            xmin=-5000,
            ymin=-5000,
            xmax=7000,
            ymax=7000,
            fig=fig,
            ax=ax,
            vmin=0,
            vmax=10,
            ret_im=True,
            title=None,
            animated=True,
        )
        algo.plugins.append(anigen)

        if args.debug:
            anigen_debug = foxes.output.SeqWakeDebugPlugin(ax=ax)
            algo.plugins.append(anigen_debug)
        
    # run all states sequentially:
    for r in algo:
        print(algo.index)

    print("\nFarm results:\n")
    print(algo.farm_results)

    if args.animation:
        print("\nCalculating animation")

        anim = foxes.output.Animator(fig)
        anim.add_generator(anigen.gen_images())
        if args.debug:
            anim.add_generator(anigen_debug.gen_images())
        ani = anim.animate(interval=600)

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
