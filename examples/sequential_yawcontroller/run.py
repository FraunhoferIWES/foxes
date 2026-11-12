## FOXES commit: c4744bd7bd3a4f163a5744fd6c5ce3b0be818a21

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV


def main(argv=None):
    """Run this tutorial with an explicit argument list or command-line options."""
    parser = argparse.ArgumentParser()
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
        print("Running example: sequential_yawcontroller")

    mbook = foxes.models.ModelBook()
    # Preserve the yaw-controller flow field, rotor indicators and labels.
    for suffix in ("gif", "mp4"):
        mbook.outputs[f"create_{suffix}"] = foxes.models.outputs.FlowAnimation(
            base_name=f"ani.{suffix}",
            fps=args.fps,
            figsize=(6.4, 4.8),
            grid_pars=dict(resolution=10),
            plot_pars=dict(
                vmin=0,
                levels=None,
                quiver_n=33,
                quiver_pars=dict(scale=0.022, alpha=0.5),
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
        kwargs = {"ref_xy": [3000, 1280], "dt_min": 1 / 6}

    states = States(
        data_source=sdata,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "WS", FV.WD: "WD"},
        fixed_vars={FV.RHO: 1.225, FV.TI: 0.07},
        states_sel=range(args.max_state) if args.max_state is not None else None,
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
        ),
        verbosity=max(args.verbosity - 1, 0),
    )
    farm.add_turbine(
        foxes.Turbine(
            xy=np.array([4170, 1280.0]),
            turbine_models=tmodels,
        ),
        verbosity=max(args.verbosity - 1, 0),
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
        ax = foxes.algorithms.Downwind(
            farm,
            foxes.input.states.SingleStateStates(ws=8, wd=270, ti=0.08, rho=1.225),
            wake_models=[],
            mbook=mbook,
            verbosity=max(args.verbosity - 1, 0),
        ).calc_outputs(
            outputs=foxes.models.outputs.FarmLayoutOutput(
                operation="get_figure", parameters=dict()
            )
        )
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
        plugins=[
            foxes.models.seq_plugins.SeqAnimationOutputPlugin(
                animations,
                debug=args.debug,
            )
        ],
        max_it=args.max_it,
        conv_crit="default" if args.max_it is None else None,
        verbosity=max(args.verbosity - 1, 0),
    )

    engine = foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
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
