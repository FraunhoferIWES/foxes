## FOXES commit: c4744bd7bd3a4f163a5744fd6c5ce3b0be818a21

import argparse
import numpy as np
import pandas as pd
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
        default=10,
    )
    parser.add_argument("--debug", help="Switch on wake debugging", action="store_true")

    parser.add_argument(
        "-S",
        "--max_state",
        help="States subset to the first n states",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-nt", "--n_turbines", help="The number of turbines", default=2, type=int
    )
    parser.add_argument(
        "-mit", "--max_it", help="Run until maximal iteration", default=None, type=int
    )
    parser.add_argument(
        "-s",
        "--states",
        help="The timeseries input file (path or static)",
        default="10s_TEST.csv",
    )
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["my_Bastankhah2016", "my_CrespoHernandez"],
        nargs="+",
    )
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes models", default=None, nargs="+"
    )
    parser.add_argument("-f", "--frame", help="The wake frame", default="seq_dyn_wakes")
    parser.add_argument(
        "-d", "--deflection", help="The wake deflection", default="Jimenez"
    )
    parser.add_argument(
        "-y",
        "--yawm",
        help="The uniform yaw misalignment value",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-yr",
        "--max_yaw_rate",
        help="The maximum yaw rate value (deg/s)",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "-ym",
        "--max_yawm",
        help="The maximum yaw misalignment value (deg)",
        type=float,
        default=7.5,
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
    parser.add_argument("-e", "--engine", help="The engine", default="NumpyEngine")
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
        default=None,
        type=int,
    )
    parser.add_argument(
        "-nf", "--nofig", help="Do not show figures", action="store_true"
    )
    parser.add_argument(
        "-b0",
        "--background0",
        help="Switch off dynamic background interpretation",
        action="store_true",
    )
    args = parser.parse_args()

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    ## Ugly manual tuning
    mbook.wake_models["my_Bastankhah2016"] = (
        foxes.models.wake_models.wind.Bastankhah2016(
            alpha=1.0, beta=0.077, k=0.04, superposition="vector"
        )
    )

    mbook.wake_models["my_CrespoHernandez"] = (
        foxes.models.wake_models.ti.CrespoHernandezTIWake(
            k=0.04, superposition="ti_max"
        )
    )

    sdata = pd.read_csv(
        foxes.StaticData().get_file_path(foxes.STATES, args.states),
        index_col=0,
        parse_dates=[0],
    )

    if args.background0:
        States = foxes.input.states.Timeseries
        kwargs = {}
    else:
        States = foxes.input.states.OnePointFlowTimeseries
        kwargs = {"ref_xy": [3000, 1280]}

    states = States(
        data_source=sdata,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "WS", FV.WD: "WD"},
        fixed_vars={FV.RHO: 1.225, FV.TI: 0.07},
        states_sel=range(args.max_state) if args.max_state is not None else None,
        dt_min=1 / 6,
        **kwargs,
    )

    mbook.turbine_models["dummy_yaw_ctrl"] = foxes.models.turbine_models.YawController(
        max_yawm=args.max_yawm,
        max_yaw_rate=args.max_yaw_rate,
    )
    tmodels = ["dummy_yaw_ctrl"] + args.tmodels + [ttype.name]

    farm = foxes.WindFarm()
    farm.add_turbine(
        foxes.Turbine(
            xy=np.array([3315.0, 1280.0]),
            turbine_models=tmodels,
        )
    )
    farm.add_turbine(
        foxes.Turbine(
            xy=np.array([4170, 1280.0]),
            turbine_models=tmodels,
        )
    )
    """
    farm.add_turbine(
        foxes.Turbine(
            xy=np.array([4170, 1280.0]),
            turbine_models=args.tmodels + [ttype.name],
        )
    )
    """

    if not args.nofig and args.show_layout:
        ax = foxes.output.FarmLayoutOutput(farm).get_figure()
        plt.show()
        plt.close(ax.get_figure())

    algo = foxes.algorithms.Sequential(
        farm,
        states=states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame=args.frame,
        wake_deflection=args.deflection,
        partial_wakes=args.pwakes,
        mbook=mbook,
        max_it=args.max_it,
        conv_crit="default" if args.max_it is None else None,
        verbosity=0,
    )

    engine = foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
    )

    # in case of animation, add a plugin that creates the images:
    if args.animation:
        anigen = foxes.output.SeqFlowAnimationPlugin(
            orientation="xy",
            var=FV.WS,
            resolution=10,
            vmin=0,
        )
        algo.plugins.append(anigen)

        if args.debug:
            anigen_debug = foxes.output.SeqWakeDebugPlugin()
            algo.plugins.append(anigen_debug)

    # run all states sequentially:
    with engine:
        for r in algo:
            print(algo.index)

    print("\nFarm results:\n")
    print(algo.farm_results)

    if not args.nofig:
        fres = algo.farm_results
        plt.plot(fres["state"], fres["WD"].to_numpy()[:, 0], label="WD turbine 0")
        plt.plot(fres["state"], fres["WD"].to_numpy()[:, 1], label="WD turbine 1")
        plt.scatter(
            fres["state"],
            fres["YAW"].to_numpy()[:, 0],
            label="YAW turbine 0",
            marker="x",
            s=15,
        )
        plt.scatter(
            fres["state"],
            fres["YAW"].to_numpy()[:, 1],
            label="YAW turbine 1",
            marker="x",
            s=15,
        )
        plt.legend()
        plt.title(
            f"max_yawm = {args.max_yawm} deg, max_yaw_rate = {args.max_yaw_rate} deg/s"
        )
        plt.xlabel("Time step")
        plt.ylabel("Degrees")
        plt.show()

    if args.animation:
        print("\nCalculating animation")

        fig, ax = plt.subplots()
        anim = foxes.output.Animator(fig)
        anim.add_generator(
            anigen.gen_images(
                ax,
                levels=None,
                quiver_pars=dict(scale=0.022, alpha=0.5),
                quiver_n=33,
                title=None,
                rotor_color="red",
            )
        )

        if args.debug:
            anim.add_generator(anigen_debug.gen_images(ax))

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

        fpath = Path(args.ani_file)
        print("Writing file", fpath)
        if fpath.suffix == ".gif":
            ani.save(filename=fpath, writer="pillow", fps=args.fps)
        else:
            ani.save(filename=fpath, writer="ffmpeg", fps=args.fps)
