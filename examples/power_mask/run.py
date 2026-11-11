import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV
from foxes.models.outputs import FarmResultsOutput


def main(argv=None):
    """Run this tutorial with an explicit argument list or command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nt", "--n_t", help="The number of turbines", type=int, default=5
    )
    parser.add_argument(
        "-s",
        "--states",
        help="The timeseries input file (path or static)",
        default="states.csv",
    )
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument(
        "-dx", "--deltax", help="The turbine distance in x", type=float, default=500.0
    )
    parser.add_argument(
        "-dy", "--deltay", help="Turbine layout y step", type=float, default=0.0
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes models", default=None, nargs="+"
    )
    parser.add_argument(
        "-f", "--pmax_file", help="The max_P csv file", default="power_mask.csv"
    )
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["Jensen_linear_k007"],
        nargs="+",
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
        print("Running example: power_mask")

    if args.verbosity > 1:
        print("\nReading file", args.pmax_file)
    cols = [f"Pmax_{i}" for i in range(args.n_t)]
    Pmax_data = pd.read_csv(args.pmax_file, usecols=cols).to_numpy()

    mbook = foxes.models.ModelBook()
    if args.nofig:
        for name in args.outputs:
            output = mbook.outputs[name]
            if hasattr(output, "show"):
                output.show = False
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype
    mbook.turbine_models["set_Pmax"] = foxes.models.turbine_models.SetFarmVars()
    mbook.turbine_models["set_Pmax"].add_var(FV.MAX_P, Pmax_data)
    models = args.tmodels + ["set_Pmax", ttype.name, "PMask"]

    engine = foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
        verbosity=max(args.verbosity - 1, 0),
    )

    if not args.nofig:
        # Compare explicit scan turbines through the ordinary selected output.
        mbook.turbine_models["curve_Pmax"] = foxes.models.turbine_models.SetFarmVars()
        mbook.turbine_models["curve_Pmax"].add_var(FV.MAX_P, 3000.0)
        scan_farm = foxes.WindFarm()
        for name, curve_models in (
            ("default", [ttype.name]),
            ("PMask", ["curve_Pmax", ttype.name, "PMask"]),
        ):
            scan_farm.add_turbine(
                foxes.Turbine([0, 0], name=name, turbine_models=curve_models),
                verbosity=max(args.verbosity - 1, 0),
            )
        scan = foxes.algorithms.Downwind(
            scan_farm,
            foxes.input.states.ScanStates(
                scans={FV.WS: np.arange(0, 30.1, 0.1)},
                fixed_vars={FV.WD: 270, FV.TI: 0.05, FV.RHO: 1.225},
            ),
            wake_models=[],
            mbook=mbook,
            verbosity=max(args.verbosity - 1, 0),
        )
        with engine:
            axs = scan.calc_farm(outputs=foxes.models.outputs.TurbineTypeCurves())
        plt.show()
        plt.close(axs[0].figure)

    states = foxes.input.states.StatesTable(
        data_source=args.states,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.WD: 270.0, FV.TI: 0.05, FV.RHO: 1.225},
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=[0.0, 0.0],
        xy_step=[args.deltax, args.deltay],
        n_turbines=args.n_t,
        turbine_models=models,
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

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame="rotor_wd",
        partial_wakes=args.pwakes,
        mbook=mbook,
        verbosity=max(args.verbosity - 1, 0),
    )

    outputs = [
        FV.D,
        FV.WD,
        FV.AMB_REWS,
        FV.REWS,
        FV.AMB_P,
        FV.P,
        FV.CT,
        FV.WEIGHT,
        FV.MAX_P,
    ]

    # Preserve the fields required by this example's farm collector.
    mbook.outputs["farm_results"] = FarmResultsOutput(variables=outputs)

    # run calculation with power mask:
    masked_results = None
    with engine:
        farm_results = algo.calc_farm(outputs=args.outputs)

        # Skip evaluation, but not the comparison run, for writer-only outputs.
        if isinstance(farm_results, dict):
            farm_results = farm_results.get("farm_results")
        elif not any(
            isinstance(mbook.outputs[name], foxes.models.outputs.FarmResultsOutput)
            for name in args.outputs
        ):
            farm_results = None
        if farm_results is not None and args.verbosity > 1:
            print("\nFarm results:\n", farm_results)
        if farm_results is not None and set(outputs).issubset(farm_results.data_vars):
            fr = farm_results.to_dataframe()
            if args.verbosity > 1:
                print(
                    fr[[FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P, FV.CT, FV.MAX_P]]
                )

            masked_results = farm_results
        else:
            print(
                "Skipping masked farm reporting: selected outputs lack the required fields."
            )

        # run calculation without power mask:
        for t in farm.turbines:
            t.models.remove("set_Pmax")
            t.models.remove("PMask")

        # The unmasked run no longer produces a maximum-power field.
        mbook.outputs["farm_results"] = FarmResultsOutput(
            variables=[variable for variable in outputs if variable != FV.MAX_P]
        )

        farm_results = algo.calc_farm(outputs=args.outputs)

    # Evaluate collected farm data only; both output runs have finished.
    if isinstance(farm_results, dict):
        farm_results = farm_results.get("farm_results")
    elif not any(
        isinstance(mbook.outputs[name], foxes.models.outputs.FarmResultsOutput)
        for name in args.outputs
    ):
        farm_results = None
    if farm_results is not None and args.verbosity > 1:
        print("\nFarm results:\n", farm_results)
    required = {FV.AMB_P, FV.AMB_REWS, FV.CT, FV.P, FV.REWS, FV.WD}
    if farm_results is None or not required.issubset(farm_results.data_vars):
        print(
            "Skipping farm reporting: selected outputs do not collect the required fields:",
            sorted(required),
        )
        return

    fr = farm_results.to_dataframe()
    if args.verbosity > 1:
        print(fr[[FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P, FV.CT]])

    if not args.nofig and masked_results is not None:
        for variable, label, cmap, maximum in (
            (FV.P, "Power", "YlOrRd", np.nanmax(Pmax_data)),
            (FV.CT, "ct", "YlGn", 1.0),
        ):
            fig, axs = plt.subplots(1, 3, figsize=(11, 5))
            for ax, data, field, title, colors, limit in (
                (
                    axs[0],
                    farm_results,
                    variable,
                    f"{label}, no power mask",
                    cmap,
                    maximum,
                ),
                (
                    axs[1],
                    masked_results,
                    FV.MAX_P,
                    "Power mask",
                    "YlOrRd",
                    np.nanmax(Pmax_data),
                ),
                (
                    axs[2],
                    masked_results,
                    variable,
                    f"{label}, with power mask",
                    cmap,
                    maximum,
                ),
            ):
                with engine:
                    algo.calc_outputs(
                        outputs=foxes.models.outputs.StateTurbineMap(
                            variable=field,
                            plot_pars=dict(
                                ax=ax,
                                title=title,
                                cmap=colors,
                                edgecolor="white",
                                vmin=0,
                                vmax=limit,
                            ),
                        ),
                        farm_results=data,
                    )
            plt.show()
            plt.close(fig)


if __name__ == "__main__":
    main()
