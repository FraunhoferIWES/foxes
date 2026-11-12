import argparse
import numpy as np
import pandas as pd

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
        default=4,
    )
    parser.add_argument(
        "-b0",
        "--background0",
        help="Switch off dynamic background interpretation",
        action="store_true",
    )
    parser.add_argument(
        "-ws0", "--fix_ws", help="Switch off dynamic wind speed", action="store_true"
    )
    parser.add_argument(
        "-S",
        "--max_state",
        help="States subset to the first n states",
        type=int,
        default=None,
    )
    parser.add_argument(
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
        "-mit", "--max_it", help="Run until maximal iteration", default=None, type=int
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
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["Bastankhah2014_linear_loc_k004"],
        nargs="+",
    )
    parser.add_argument(
        "-wl",
        "--wake_length_km",
        help="The maximal wake length in km (for applicable wake models)",
        type=float,
        default=8.0,
    )
    parser.add_argument("-f", "--frame", help="The wake frame", default="dyn_wakes")
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
        default=None,
        type=int,
    )
    parser.add_argument(
        "-nf", "--nofig", help="Do not show figures", action="store_true"
    )

    # Select chunk outputs by their model-book names.
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
        print("Running example: dyn_wakes")

    mbook = foxes.models.ModelBook()
    mbook.outputs["farm_results"].fdata_variables.extend([FV.CT, FV.EFF])

    # Customize only this example's named animation outputs.
    for output, suffix in (
        ("create_turbine_gif", "gif"),
        ("create_turbine_mp4", "mp4"),
    ):
        mbook.outputs[output] = foxes.models.outputs.FlowTurbineAnimation(
            turbines=[1, 4, 7],
            ref_xy=args.ref_xy,
            base_name=f"ani.{suffix}",
            fps=args.fps,
            figsize=(5.2, 7),
            grid_pars=dict(resolution=30, xmax=5000, ymax=5000),
            plot_pars=dict(
                vmin=0,
                quiver_pars=dict(angles="xy", scale_units="xy", scale=0.013),
                quiver_n=35,
                title=None,
                rotor_color="red",
            ),
            layout_pars=dict(
                title="", annotate=1, anno_delx=-120, anno_dely=-60, alpha=0
            ),
        )

    if args.show_layout:
        args.outputs.insert(0, "farm_layout")
    # Configure outputs once; figure suppression is handled by the framework.
    for name in [*args.results, *args.outputs]:
        output = mbook.outputs[name]
        output.nofig = args.nofig
        if isinstance(output, foxes.models.outputs.FlowAnimation):
            output.verbosity = args.verbosity

    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    # optionally set turbines in yaw:
    N = int(args.n_turbines**0.5)
    if args.yawm is None:
        ymodels = []
    else:
        yawm = np.zeros((1, N * N), dtype=np.float64)
        yawm[:, :N] = args.yawm
        mbook.turbine_models["set_yawm"] = foxes.models.turbine_models.SetFarmVars(
            pre_rotor=True
        )
        mbook.turbine_models["set_yawm"].add_var(FV.YAWM, yawm)
        ymodels = ["set_yawm"]

    if args.background0:
        States = foxes.input.states.Timeseries
        kwargs = {}
    else:
        States = foxes.input.states.OnePointFlowTimeseries
        kwargs = {"ref_xy": args.ref_xy}

    sdata = pd.read_csv(
        foxes.StaticData().get_file_path(foxes.STATES, args.states),
        index_col=0,
        parse_dates=[0],
    )
    if not args.fix_ws:
        n_times = len(sdata.index)
        sdata["ws"] = 5 + 0.3 * np.sin(np.arange(n_times) * 2 * np.pi / 20)

    states = States(
        data_source=sdata,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
        fixed_vars={FV.RHO: 1.225, FV.TI: 0.07},
        states_sel=range(args.max_state) if args.max_state is not None else None,
        **kwargs,
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

    algo = foxes.algorithms.Iterative(
        farm,
        states=states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame=args.frame,
        wake_deflection=args.deflection,
        partial_wakes=args.pwakes,
        max_wake_length_km=args.wake_length_km,
        mbook=mbook,
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

    for variable in (FV.CT, FV.AMB_WD):
        if variable not in mbook.outputs["farm_results"].fdata_variables:
            mbook.outputs["farm_results"].fdata_variables.append(variable)
    with engine:
        # Preserve every result key, including single-output selections.
        results = algo.calc_farm(outputs=args.results, ret_dict=True)
        if args.verbosity > 1:
            for key, data in results.items():
                print(f"\n{key}:\n{data}")

        # Process each output separately so different point grids remain independent.
        for name in args.outputs:
            output_results = algo.calc_outputs(
                outputs=name, farm_results=results.get("farm_results"), ret_dict=True
            )
            if args.verbosity > 1:
                for key, data in output_results.items():
                    print(f"\n{key}:\n{data}")


if __name__ == "__main__":
    main()
