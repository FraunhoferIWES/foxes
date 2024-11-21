import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import foxes
import foxes.variables as FV


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--animation", help="Write flow animation file", action="store_true"
    )
    parser.add_argument(
        "-A",
        "--ani_file",
        help="Path to the animation file to be written",
        default="ani.gif",
    )
    parser.add_argument(
        "-F",
        "--fps",
        help="The frames per second value for the animation",
        type=int,
        default=4,
    )
    parser.add_argument(
        "-b",
        "--background",
        help="Switch on dynamic background interpretation",
        action="store_true",
    )
    parser.add_argument(
        "-S",
        "--max_state",
        help="States subset to the first n states",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-R",
        "--ref_xy",
        help="The reference point x y",
        type=float,
        default=[2500, 2500],
        nargs="+",
    )
    parser.add_argument(
        "-nt", "--n_turbines", help="The number of turbines", default=9, type=int
    )
    parser.add_argument(
        "-s",
        "--states",
        help="The timeseries input file (path or static)",
        default="timeseries_100.csv.gz",
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
    parser.add_argument("-sc", "--scheduler", help="The scheduler choice", default=None)
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["Bastankhah2014_linear_lim_k004"],
        nargs="+",
    )
    parser.add_argument("-f", "--frame", help="The wake frame", default="timelines")
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=[], nargs="+"
    )
    parser.add_argument(
        "-sl",
        "--show_layout",
        help="Flag for showing layout figure",
        action="store_true",
    )
    parser.add_argument("-e", "--engine", help="The engine", default="process")
    parser.add_argument(
        "-n", "--n_cpus", help="The number of cpus", default=None, type=int
    )
    parser.add_argument(
        "-c",
        "--chunksize_states",
        help="The chunk size for states",
        default=20,
        type=int,
    )
    parser.add_argument(
        "-C",
        "--chunksize_points",
        help="The chunk size for points",
        default=5000,
        type=int,
    )
    parser.add_argument(
        "-nf", "--nofig", help="Do not show figures", action="store_true"
    )
    args = parser.parse_args()

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    if not args.background:
        States = foxes.input.states.Timeseries
        kwargs = {}
    else:
        States = foxes.input.states.OnePointFlowTimeseries
        kwargs = {"ref_xy": args.ref_xy}

    states = States(
        data_source=args.states,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
        fixed_vars={FV.RHO: 1.225, FV.TI: 0.07},
        states_sel=range(args.max_state) if args.max_state is not None else None,
        **kwargs,
    )

    farm = foxes.WindFarm()
    N = int(args.n_turbines**0.5)
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
        plt.close(ax.get_figure())

    algo = foxes.algorithms.Iterative(
        farm,
        states=states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame=args.frame,
        partial_wakes=args.pwakes,
        mbook=mbook,
        engine=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
        verbosity=1,
    )

    time0 = time.time()
    farm_results = algo.calc_farm()
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
    print(farm_df[[FV.AMB_REWS, FV.REWS, FV.CT, FV.EFF]].describe())

    # power results
    P0 = o.calc_mean_farm_power(ambient=True)
    P = o.calc_mean_farm_power()
    print(f"\nFarm power        : {P/1000:.1f} MW")
    print(f"Farm ambient power: {P0/1000:.1f} MW")
    print(f"Farm efficiency   : {o.calc_farm_efficiency()*100:.2f} %")

    if not args.nofig:
        sts = np.arange(farm_results.sizes["state"])
        plt.plot(sts, farm_results.REWS[:, 4], label="Turbine 4")
        plt.plot(sts, farm_results.REWS[:, 7], label="Turbine 7")
        plt.legend()
        plt.xlabel("State")
        plt.ylabel("REWS [m/s]")
        plt.show()

    if not args.nofig and args.animation:
        print("\nCalculating animation")

        fig, axs = plt.subplots(
            2, 1, figsize=(5.2, 7), gridspec_kw={"height_ratios": [3, 1]}
        )

        anim = foxes.output.Animator(fig)
        of = foxes.output.FlowPlots2D(algo, farm_results)
        anim.add_generator(
            of.gen_states_fig_xy(
                FV.WS,
                resolution=30,
                quiver_pars=dict(angles="xy", scale_units="xy", scale=0.013),
                quiver_n=35,
                xmax=5000,
                ymax=5000,
                vmin=0,
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

        fpath = Path(args.ani_file)
        print("Writing file", fpath)
        if fpath.suffix == ".gif":
            ani.save(filename=fpath, writer="pillow", fps=args.fps)
        else:
            ani.save(filename=fpath, writer="ffmpeg", fps=args.fps)
