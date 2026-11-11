import time
import argparse
import numpy as np
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
        default=8.0,
    )
    parser.add_argument("-f", "--frame", help="The wake frame", default="timelines")
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
        default=5000,
        type=int,
    )
    parser.add_argument(
        "-nf", "--nofig", help="Do not show figures", action="store_true"
    )

    # Select chunk outputs by their model-book names.
    parser.add_argument(
        "-o",
        "--outputs",
        help="The output models",
        nargs="+",
        default=["farm_results", "create_turbine_gif"],
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
        print("Running example: timelines")

    mbook = foxes.models.ModelBook()
    mbook.outputs["farm_results"].fdata_variables.extend([FV.EFF])

    # Preserve the two-panel animation through the named output models.
    for suffix in ("gif", "mp4"):
        mbook.outputs[f"create_turbine_{suffix}"] = (
            foxes.models.outputs.FlowTurbineAnimation(
                base_name=f"ani.{suffix}",
                turbines=[1, 4, 7],
                ref_xy=args.ref_xy,
                fps=args.fps,
                figsize=(5.2, 7),
                grid_pars=dict(resolution=30, xmax=5000, ymax=5000),
                plot_pars=dict(
                    vmin=0,
                    quiver_n=35,
                    title=None,
                    rotor_color="red",
                    quiver_pars=dict(angles="xy", scale_units="xy", scale=0.013),
                ),
                layout_pars=dict(
                    title="", annotate=1, anno_delx=-120, anno_dely=-60, alpha=0
                ),
            )
        )

    # Keep animation diagnostics at the script's verbosity level.
    for name in args.outputs:
        selected = mbook.outputs[name]
        if isinstance(selected, foxes.models.outputs.FlowAnimation):
            selected.verbosity = args.verbosity
        if args.nofig and hasattr(selected, "show"):
            selected.show = False

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
                operation="get_figure", parameters=dict()
            )
        )
        plt.show()
        plt.close(ax.get_figure())

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
        verbosity=max(args.verbosity - 1, 0),
    )

    engine = foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
        verbosity=max(args.verbosity - 1, 0),
    )

    with engine:
        time0 = time.time()
        # Request per-state metrics through output selection.
        farm_results = algo.calc_farm(outputs=args.outputs)
        time1 = time.time()

    # Evaluate collected farm data only; writer-only outputs are already complete.
    if isinstance(farm_results, dict):
        farm_results = farm_results.get("farm_results")
    elif not any(
        isinstance(mbook.outputs[name], foxes.models.outputs.FarmResultsOutput)
        for name in args.outputs
    ):
        farm_results = None
    if farm_results is not None and args.verbosity > 1:
        print("\nFarm results:\n", farm_results)
    required = {
        FV.AMB_P,
        FV.AMB_REWS,
        FV.AMB_TI,
        FV.CT,
        FV.EFF,
        FV.P,
        FV.REWS,
        FV.TI,
        FV.WD,
        FV.X,
        FV.Y,
    }
    if farm_results is None or not required.issubset(farm_results.data_vars):
        print(
            "Skipping farm reporting: selected outputs do not collect the required fields:",
            sorted(required),
        )
        return

    if args.verbosity > 1:
        print("\nCalc time =", time1 - time0, "\n")

    # state-turbine results
    farm_df = farm_results.to_dataframe()
    if args.verbosity > 1:
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

    if not args.nofig:
        sts = np.arange(farm_results.sizes["state"])
        plt.plot(sts, farm_results.REWS[:, 4], label="Turbine 4")
        plt.plot(sts, farm_results.REWS[:, 7], label="Turbine 7")
        plt.legend()
        plt.xlabel("State")
        plt.ylabel("REWS [m/s]")
        plt.show()


if __name__ == "__main__":
    main()
