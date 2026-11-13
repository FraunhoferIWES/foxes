import argparse
import matplotlib.pyplot as plt
import numpy as np

import foxes
import foxes.variables as FV


def main(argv=None):
    """Run this tutorial with an explicit argument list or command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-state", type=int, default=240)
    parser.add_argument(
        "-F",
        "--fps",
        help="The frames per second value for the animation",
        type=int,
        default=4,
    )
    parser.add_argument("--debug", help="Switch on wake debugging", action="store_true")
    parser.add_argument(
        "-S",
        "--n_states",
        help="The number of states",
        type=int,
        default=50,
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
        "-wl",
        "--wake_length_km",
        help="The maximal wake length in km (for applicable wake models)",
        type=float,
        default=None,
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
        "-V",
        "--variable",
        help="Plot variable",
        default=FV.WS,
    )
    parser.add_argument(
        "-VMAX",
        "--max_variable",
        help="Maximal plot variable",
        default=10,
        type=float,
    )
    parser.add_argument("-e", "--engine", help="The engine", default="NumpyEngine")
    parser.add_argument(
        "-d", "--deflection", help="The wake deflection", default="no_deflection"
    )
    parser.add_argument(
        "-y",
        "--yawm",
        help="The uniform yaw misalignment value",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-n", "--n_cpus", help="The number of cpus", default=None, type=int
    )
    parser.add_argument(
        "-c",
        "--chunksize_states",
        help="The chunk size for states",
        default=None,
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
        "-R",
        "--results",
        help="The calculation result models",
        nargs="+",
        default=["farm_results"],
    )
    parser.add_argument(
        "-o",
        "--outputs",
        help="The post-processing output models",
        nargs="+",
        default=["create_turbine_gif"],
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=2,
        help="0 quiet, 1 summary, 2 reports, 3 library progress",
    )
    args = parser.parse_args(argv)
    if args.verbosity > 0:
        print("Running example: sequential")

    mbook = foxes.models.ModelBook()
    # Keep flow-only output content and settings local to this example.
    for suffix in ("gif", "mp4"):
        mbook.outputs[f"create_{suffix}"] = foxes.models.outputs.FlowAnimation(
            base_name=f"ani.{suffix}",
            variables=[args.variable],
            fps=args.fps,
            figsize=(6.4, 4.8),
            grid_pars=dict(resolution=10, xmin=-1000, ymin=-1000, xmax=4000, ymax=3000),
            plot_pars=dict(
                vmin=0,
                vmax=args.max_variable,
                levels=None,
                quiver_n=157,
                quiver_pars=dict(scale=0.016, alpha=0.5),
                title=None,
                rotor_color="red",
            ),
            layout_pars=dict(
                title="", annotate=1, anno_delx=-120, anno_dely=-60, alpha=0, s=10
            ),
        )

    # Keep animation diagnostics at the script's verbosity level.
    animations = []
    for name in dict.fromkeys([*args.results, *args.outputs]):
        selected = mbook.outputs[name]
        selected.nofig = args.nofig
        if isinstance(selected, foxes.models.outputs.FlowAnimation):
            selected.verbosity = args.verbosity
            animations.append(name)
        elif not isinstance(selected, foxes.models.outputs.FarmResultsOutput):
            parser.error(
                f"{name}: Sequential examples support farm outputs and animations"
            )

    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    # optionally set turbines in yaw:
    N = 3
    if args.yawm is None:
        ymodels = []
    else:
        yawm = np.zeros((args.n_states, N * N), dtype=np.float64)
        yawm[:, :N] = args.yawm
        mbook.turbine_models["set_yawm"] = foxes.models.turbine_models.SetFarmVars(
            pre_rotor=True
        )
        mbook.turbine_models["set_yawm"].add_var(FV.YAWM, yawm)
        ymodels = ["set_yawm"]

    states = foxes.input.states.Timeseries(
        data_source="timeseries_3000.csv.gz",
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        # fixed_vars={FV.WD: 270},
        var2col={FV.WD: "WD", FV.WS: "WS", FV.TI: "TI", FV.RHO: "RHO"},
        states_sel=range(args.start_state, args.start_state + args.n_states),
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_grid(
        farm,
        xy_base=np.array([0.0, 0.0]),
        step_vectors=np.array([[1000.0, 0], [0, 800.0]]),
        steps=(N, N),
        turbine_models=ymodels + args.tmodels + [ttype.name],
        verbosity=max(args.verbosity - 1, 0),
    )

    if not args.nofig and args.show_layout:
        ax = foxes.algorithms.Downwind(
            farm,
            foxes.input.states.SingleStateStates(ws=8, wd=270, ti=0.08, rho=1.225),
            wake_models=[],
            mbook=mbook,
            verbosity=max(args.verbosity - 1, 0),
        ).calc_outputs(
            outputs=foxes.models.outputs.FarmLayoutOutput(
                operation="get_figure", parameters={"figsize": (8, 8)}
            )
        )
        plt.show()
        plt.close(ax.figure)

    engine = foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
        verbosity=max(args.verbosity - 1, 0),
    )

    algo = foxes.algorithms.Sequential(
        farm,
        states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame=args.frame,
        max_wake_length_km=args.wake_length_km,
        wake_deflection=args.deflection,
        partial_wakes=args.pwakes,
        mbook=mbook,
        plugins=[
            foxes.models.seq_plugins.SeqAnimationOutputPlugin(
                animations,
                debug=args.debug,
            )
        ],
        verbosity=max(args.verbosity - 1, 0),
    )

    # run all states sequentially:
    with engine:
        for r in algo:
            if args.verbosity > 1:
                print(algo.index)

    # Reuse the accumulated iterator data for collector and farm-only outputs.
    processor = foxes.algorithms.Downwind(
        farm,
        states,
        wake_models=[],
        mbook=mbook,
        verbosity=max(args.verbosity - 1, 0),
    )
    with engine:
        result_models = [
            mbook.outputs[name]
            for name in args.results
            if not isinstance(mbook.outputs[name], foxes.models.outputs.FlowAnimation)
        ]
        results = processor.calc_outputs(
            outputs=result_models, farm_results=algo.farm_results, ret_dict=True
        )
        if args.verbosity > 1:
            for key, data in results.items():
                print(f"\n{key}:\n{data}")
        for name in args.outputs:
            output = mbook.outputs[name]
            if isinstance(output, foxes.models.outputs.FlowAnimation):
                continue
            output_results = processor.calc_outputs(
                outputs=name, farm_results=results.get("farm_results"), ret_dict=True
            )
            if args.verbosity > 1:
                for key, data in output_results.items():
                    print(f"\n{key}:\n{data}")


if __name__ == "__main__":
    main()
