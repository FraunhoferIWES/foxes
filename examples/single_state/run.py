import numpy as np
import argparse
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV


def main(argv=None):
    """Run this tutorial with an explicit argument list or command-line options."""
    # define arguments and options:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", default=None)
    parser.add_argument(
        "-nt", "--n_t", help="The number of turbines", type=int, default=5
    )
    parser.add_argument(
        "-l",
        "--layout",
        help="The wind farm layout file (path or static)",
        default=None,
    )
    parser.add_argument("--ws", help="The wind speed", type=float, default=9.0)
    parser.add_argument("--wd", help="The wind direction", type=float, default=270.0)
    parser.add_argument("--ti", help="The TI value", type=float, default=0.08)
    parser.add_argument("--rho", help="The air density", type=float, default=1.225)
    parser.add_argument(
        "-dx", "--deltax", help="The turbine distance in x", type=float, default=800.0
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
        "-m",
        "--tmodels",
        help="The turbine models",
        default=[],
        nargs="+",
    )
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["Bastankhah2014_linear_k004", "CrespoHernandez_quadratic_ambka04"],
        nargs="+",
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes models", default=None, nargs="+"
    )
    parser.add_argument(
        "-g", "--grounds", help="The ground models", default=None, nargs="+"
    )
    parser.add_argument("-f", "--frame", help="The wake frame", default="rotor_wd")
    parser.add_argument("--var", help="The plot variable", default=FV.WS)
    parser.add_argument(
        "-it", "--iterative", help="Use iterative algorithm", action="store_true"
    )
    parser.add_argument(
        "-nf", "--nofig", help="Do not show figures", action="store_true"
    )

    # Select chunk outputs by their model-book names.
    parser.add_argument(
        "-o", "--outputs", help="The output models", nargs="+", default=["farm_results"]
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
        print("Running example: single_state")

    # create model book
    mbook = foxes.ModelBook()
    mbook.outputs["farm_results"].fdata_variables.extend(
        [FV.AMB_WD, FV.CT, FV.EFF, FV.ORDER]
    )

    if args.nofig:
        for name in args.outputs:
            output = mbook.outputs[name]
            if hasattr(output, "show"):
                output.show = False
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    # create states
    states = foxes.input.states.SingleStateStates(
        ws=args.ws,
        wd=args.wd,
        ti=args.ti,
        rho=args.rho,
    )

    # create wind farm
    if args.verbosity > 1:
        print("\nCreating wind farm")
    farm = foxes.WindFarm()
    if args.layout is None:
        foxes.input.farm_layout.add_row(
            farm=farm,
            xy_base=np.array([0.0, 0.0]),
            xy_step=np.array([args.deltax, args.deltay]),
            n_turbines=args.n_t,
            turbine_models=args.tmodels + [ttype.name],
            verbosity=max(args.verbosity - 1, 0),
        )
    else:
        foxes.input.farm_layout.add_from_file(
            farm,
            args.layout,
            turbine_models=args.tmodels + [ttype.name],
            verbosity=max(args.verbosity - 1, 0),
        )

    # create algorithm
    Algo = foxes.algorithms.Iterative if args.iterative else foxes.algorithms.Downwind
    algo = Algo(
        farm,
        states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame=args.frame,
        partial_wakes=args.pwakes,
        ground_models=args.grounds,
        mbook=mbook,
        verbosity=max(args.verbosity - 1, 0),
    )

    engine = foxes.Engine.new(args.engine, verbosity=max(args.verbosity - 1, 0))
    with engine:
        farm_results = algo.calc_farm(outputs=args.outputs)

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
        FV.ORDER,
        FV.P,
        FV.REWS,
        FV.TI,
        FV.WD,
        FV.X,
    }
    if farm_results is None or not required.issubset(farm_results.data_vars):
        print(
            "Skipping farm reporting: selected outputs do not collect the required fields:",
            sorted(required),
        )
        return

    # state-turbine results
    farm_df = farm_results.to_dataframe()
    if args.verbosity > 1:
        print("\nFarm results data:\n")
        print(
            farm_df[
                [
                    FV.X,
                    FV.ORDER,
                    FV.WD,
                    FV.AMB_REWS,
                    FV.REWS,
                    FV.AMB_TI,
                    FV.TI,
                    FV.AMB_P,
                    FV.P,
                    FV.CT,
                    FV.EFF,
                ]
            ]
        )
        print()

    if not args.nofig:
        # horizontal flow plot

        with engine:
            figures = algo.calc_outputs(
                outputs=foxes.models.outputs.FlowPlots2D(
                    plane="xy",
                    reduction="states",
                    plot=True,
                    data_format="numpy",
                    var=args.var,
                    grid_pars=dict(resolution=10),
                    plot_pars=dict(rotor_color="red", figsize=(10, 3)),
                ),
                farm_results=farm_results,
            )
        plt.show()
        for figure in figures:
            plt.close(figure)

        # vertical flow plot

        with engine:
            figures = algo.calc_outputs(
                outputs=foxes.models.outputs.FlowPlots2D(
                    plane="xz",
                    reduction="states",
                    plot=True,
                    data_format="numpy",
                    var=args.var,
                    grid_pars=dict(x_direction=np.mod(args.wd, 360.0), resolution=20),
                    plot_pars=dict(rotor_color="red", figsize=(10, 3)),
                ),
                farm_results=farm_results,
            )
        plt.show()
        for figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    main()
