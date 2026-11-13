import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV
from foxes.models.outputs import FarmResultsOutput, StateTurbineMap, TurbineTypeCurves


def main(argv=None):
    """Compare masked and unmasked turbine curves, power and thrust maps.

    Parameters
    ----------
    argv
        Command-line arguments, or ``None`` to read the process arguments.
        ``--nofig`` suppresses all figures while retaining numerical outputs.
    """
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
        default=["state_turbine_map"],
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
            axs = scan.calc_farm(
                outputs=TurbineTypeCurves(
                    plot_pars={
                        "titles": {
                            FV.P: f"Power curve, {ttype.name}",
                            FV.CT: f"Thrust curve, {ttype.name}",
                        },
                        "y_labels": {FV.P: "Power [kW]", FV.CT: "ct [-]"},
                        "linestyle": "-",
                    }
                )
            )
        for ax in axs:
            ax.grid()
            for line, width in zip(ax.lines, [2.5, 1.8]):
                line.set_linewidth(width)
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

    # Compose shared panels while each case still owns its calculated fields.
    comparison_figures = []
    comparison_maps = {"masked": [], "unmasked": []}
    if not args.nofig and "state_turbine_map" in args.outputs:
        power_limit = np.nanmax(Pmax_data)
        for variable, label, cmap, limit in (
            (FV.P, "Power", "YlOrRd", power_limit),
            (FV.CT, "ct", "YlGn", 1.0),
        ):
            figure, axes = plt.subplots(1, 3, figsize=(11, 5))
            comparison_figures.append(figure)
            for case, ax, field, title, palette, maximum in (
                (
                    "unmasked",
                    axes[0],
                    variable,
                    f"{label}, no power mask",
                    cmap,
                    limit,
                ),
                ("masked", axes[1], FV.MAX_P, "Power mask", "YlOrRd", power_limit),
                (
                    "masked",
                    axes[2],
                    variable,
                    f"{label}, with power mask",
                    cmap,
                    limit,
                ),
            ):
                comparison_maps[case].append(
                    StateTurbineMap(
                        variable=field,
                        plot_pars={
                            "ax": ax,
                            "title": title,
                            "edgecolor": "white",
                            "cmap": palette,
                            "vmin": 0,
                            "vmax": maximum,
                            "cbar_label": "",
                        },
                    )
                )
        mbook.outputs["state_turbine_map"] = foxes.core.OutputList(
            outputs=comparison_maps["masked"]
        )

    # Evaluate the masked and unmasked cases independently.
    if args.show_layout:
        args.outputs.insert(0, "farm_layout")
    # Suppress figure creation in both stages, not just display.
    for name in [*args.results, *args.outputs]:
        output = mbook.outputs[name]
        output.nofig = args.nofig

    with engine:
        # Keep result keys even when only one model is selected.
        results = algo.calc_farm(outputs=args.results, ret_dict=True)
        if args.verbosity > 1:
            for key, data in results.items():
                print(f"\n{key}:\n{data}")

        # Independent output calls allow different point grids.
        output_results = {}
        for name in args.outputs:
            data = algo.calc_outputs(
                outputs=name, farm_results=results.get("farm_results"), ret_dict=True
            )
            if args.verbosity > 1:
                for key, value in data.items():
                    print(f"\n{key}:\n{value}")
            output_results.update(data)

        # Reuse the comparison axes for the independent unmasked calculation.
        if comparison_figures:
            mbook.outputs["state_turbine_map"] = foxes.core.OutputList(
                outputs=comparison_maps["unmasked"]
            )
        if args.show_layout:
            args.outputs.pop(0)
        for turbine in farm.turbines:
            turbine.models.remove("set_Pmax")
            turbine.models.remove("PMask")
        mbook.outputs["farm_results"] = FarmResultsOutput(
            variables=[variable for variable in outputs if variable != FV.MAX_P],
            nofig=args.nofig,
        )
        # Keep result keys even when only one model is selected.
        results = algo.calc_farm(outputs=args.results, ret_dict=True)
        if args.verbosity > 1:
            for key, data in results.items():
                print(f"\n{key}:\n{data}")

        # Independent output calls allow different point grids.
        output_results = {}
        for name in args.outputs:
            data = algo.calc_outputs(
                outputs=name, farm_results=results.get("farm_results"), ret_dict=True
            )
            if args.verbosity > 1:
                for key, value in data.items():
                    print(f"\n{key}:\n{value}")
            output_results.update(data)

    # Display only complete comparisons, then release example-owned figures.
    if comparison_figures:
        for figure in comparison_figures:
            for ax in figure.axes[:3]:
                ax.set_ylabel("Turbine index")
        plt.show()
        for figure in comparison_figures:
            plt.close(figure)


if __name__ == "__main__":
    main()
