import argparse

import foxes
import foxes.variables as FV
from foxes.models.outputs import FarmResultsOutput


def main(argv=None):
    """Run this tutorial with an explicit argument list or command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--layout",
        help="The wind farm layout file (path or static)",
        default="test_farm_67.csv",
    )
    parser.add_argument(
        "-s",
        "--states",
        help="The states input file (path or static)",
        default="wind_rose_bremen.csv",
    )
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument(
        "-sm",
        "--sector_file",
        help="The sector management file",
        default="sector_rules.csv",
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes models", default=None, nargs="+"
    )
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["Bastankhah2014_linear_k002"],
        nargs="+",
    )
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=[], nargs="+"
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
        default=["power_rose_0", "power_rose_1"],
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
        print("Running example: sector_management")

    mbook = foxes.models.ModelBook()
    mbook.outputs["farm_results"].fdata_variables.extend([FV.MAX_P])

    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    mbook.turbine_models["sector_rules"] = foxes.models.turbine_models.SectorManagement(
        data_source=args.sector_file,
        col_tnames="tname",
        range_vars=[FV.WD, FV.REWS],
        target_vars=[FV.MAX_P],
        colmap={"REWS_min": "WS_min", "REWS_max": "WS_max"},
    )

    states = foxes.input.states.StatesTable(
        data_source=args.states,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.WEIGHT: "weight"},
        fixed_vars={FV.RHO: 1.225, FV.TI: 0.05},
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=[0.0, 0.0],
        xy_step=[600.0, 0.0],
        n_turbines=2,
        turbine_models=args.tmodels + [ttype.name, "sector_rules", "PMask"],
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
        FV.AMB_WD,
        FV.H,
        FV.AMB_REWS,
        FV.REWS,
        FV.AMB_P,
        FV.MAX_P,
        FV.P,
        FV.CT,
        FV.WEIGHT,
    ]

    # Preserve the fields required by this example's farm collector.
    mbook.outputs["farm_results"] = FarmResultsOutput(variables=outputs)

    engine = foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
        verbosity=max(args.verbosity - 1, 0),
    )

    for turbine in (0, 1):
        mbook.outputs[f"power_rose_{turbine}"] = foxes.models.outputs.RosePlotOutput(
            wd_sectors=16,
            ws_var=FV.P,
            wd_var=FV.WD,
            ws_bins=[0, 100, 1000, 2000, 4000, 5001, 7000],
            turbine=turbine,
            show=True,
            plot_pars=dict(title=f"Power turbine {turbine}"),
        )
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
