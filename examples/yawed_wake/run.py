import numpy as np
import argparse

import foxes
import foxes.variables as FV


def main(argv=None):
    """Run this tutorial with an explicit argument list or command-line options."""
    # define arguments and options:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y", "--yawm", help="YAWM angle of first turbine", default=30.0, type=float
    )
    parser.add_argument(
        "-y2", "--yawm2", help="YAWM angle of second turbine", default=0.0, type=float
    )
    parser.add_argument("--ws", help="The wind speed", type=float, default=9.0)
    parser.add_argument("--wd", help="The wind direction", type=float, default=270.0)
    parser.add_argument("--ti", help="The TI value", type=float, default=0.08)
    parser.add_argument("--rho", help="The air density", type=float, default=1.225)
    parser.add_argument(
        "-dx", "--deltax", help="The turbine distance in x", type=float, default=1500.0
    )
    parser.add_argument(
        "-dy", "--deltay", help="Turbine layout y step", type=float, default=0.0
    )
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["TurbOPark_vector_ambka004", "CrespoHernandez_quadratic_ambka04"],
        nargs="+",
    )
    parser.add_argument(
        "-d", "--deflection", help="The wake deflection", default="Jimenez"
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes models", default=None, nargs="+"
    )
    parser.add_argument("-f", "--frame", help="The wake frame", default="rotor_wd")
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=[], nargs="+"
    )
    parser.add_argument("--var", help="The plot variable", default=FV.WS)
    parser.add_argument(
        "-it", "--iterative", help="Use iterative algorithm", action="store_true"
    )
    parser.add_argument("-e", "--engine", help="The engine", default="process")
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
        default=5000,
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
        default=["flow_plot_xy_states", "flow_plot_yz_states"],
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
        print("Running example: yawed_wake")

    # create model book
    mbook = foxes.ModelBook()
    mbook.outputs["farm_results"].fdata_variables.extend([FV.AMB_WD, FV.YAWM])

    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    # set turbines in yaw
    yawm = np.array([[args.yawm, args.yawm2]])
    mbook.turbine_models["set_yawm"] = foxes.models.turbine_models.SetFarmVars()
    mbook.turbine_models["set_yawm"].add_var(FV.YAWM, yawm)

    # create states
    states = foxes.input.states.SingleStateStates(
        ws=args.ws, wd=args.wd, ti=args.ti, rho=args.rho
    )

    # create wind farm
    if args.verbosity > 1:
        print("\nCreating wind farm")
    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=np.array([0.0, 0.0]),
        xy_step=np.array([args.deltax, args.deltay]),
        n_turbines=2,
        turbine_models=args.tmodels + ["set_yawm", "yawm2yaw", ttype.name],
        verbosity=max(args.verbosity - 1, 0),
    )

    # create algorithm
    Algo = foxes.algorithms.Iterative if args.iterative else foxes.algorithms.Downwind
    algo = Algo(
        farm,
        states=states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame=args.frame,
        wake_deflection=args.deflection,
        partial_wakes=args.pwakes,
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

    mbook.outputs["flow_plot_xy_states"] = foxes.models.outputs.FlowPlots2D(
        plane="xy",
        reduction="states",
        var=args.var,
        show=True,
        grid_pars=dict(resolution=10, xmin=-500, xmax=3000),
        plot_pars=dict(rotor_color="red", cmap="BuPu" if args.var == FV.WD else None),
    )
    mbook.outputs["flow_plot_yz_states"] = foxes.models.outputs.FlowPlots2D(
        plane="yz",
        reduction="states",
        var=args.var,
        show=True,
        grid_pars=dict(resolution=5, x=1500, ymin=-200, ymax=200, zmin=0, zmax=250),
        plot_pars=dict(rotor_color="red", cmap="BuPu" if args.var == FV.WD else None),
    )
    for variable in (FV.CT, FV.AMB_WD):
        if variable not in mbook.outputs["farm_results"].fdata_variables:
            mbook.outputs["farm_results"].fdata_variables.append(variable)
    # Apply figure suppression to both calculation stages.
    for name in [*args.results, *args.outputs]:
        output = mbook.outputs[name]
        output.nofig = args.nofig

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
