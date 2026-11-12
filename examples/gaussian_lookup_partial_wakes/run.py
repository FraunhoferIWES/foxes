import argparse
import foxes
import foxes.variables as FV


def main(argv=None) -> None:
    """Run this tutorial with an explicit argument list or command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--wake-model",
        help="Wake model name from model book",
        default="Bastankhah2014_linear_k004",
    )
    parser.add_argument(
        "-p",
        "--partial-wakes",
        help="Partial wakes model name from model book",
        default="gaussian_lookup",
    )
    parser.add_argument(
        "-minw",
        "--min-weight",
        help="Minimum lookup weight kept by gaussian_lookup (smaller values are zeroed)",
        type=float,
        default=1.0e-8,
    )
    parser.add_argument(
        "-bp",
        "--bounds-policy",
        help="Out-of-range policy for gaussian_lookup",
        choices=("clip", "nan", "raise"),
        default="clip",
    )
    parser.add_argument(
        "-lf",
        "--lookup-file",
        help="Load this Gaussian lookup NetCDF file into gaussian_lookup",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-r",
        "--rotor",
        help="Rotor model name from model book",
        default="centre",
    )
    parser.add_argument(
        "-l",
        "--layout",
        help="Farm layout file (path or static data key)",
        default="test_farm_67.csv",
    )
    parser.add_argument(
        "-t",
        "--timeseries",
        help="Timeseries file (path or static data key)",
        default="timeseries_3000.csv.gz",
    )
    parser.add_argument(
        "-tt",
        "--turbine-type",
        help="Turbine type from model book",
        default="NREL5MW",
    )
    parser.add_argument("--var", help="Flow plot variable", default=FV.WS)
    parser.add_argument(
        "-fsi",
        "--fig-state-index",
        help="State index used for flow plotting",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-fr",
        "--fig-resolution",
        help="Flow plot grid resolution in meters",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "-e",
        "--engine",
        help="Engine type",
        default="ProcessEngine",
    )
    parser.add_argument(
        "-n",
        "--n-cpus",
        help="Number of worker processes",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-c",
        "--chunk-size-states",
        help="State chunk size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-C",
        "--chunk-size-points",
        help="Point chunk size",
        type=int,
        default=None,
    )
    parser.add_argument("-nf", "--nofig", help="Skip figures", action="store_true")

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
        default=["flow_plot_xy_states"],
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
        print("Running example: gaussian_lookup_partial_wakes")

    states = foxes.input.states.Timeseries(
        args.timeseries,
        [FV.WS, FV.WD, FV.TI, FV.RHO],
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_from_file(
        farm,
        args.layout,
        turbine_models=[args.turbine_type],
        verbosity=max(args.verbosity - 1, 0),
    )

    mbook = foxes.models.ModelBook()
    mbook.outputs["farm_results"].fdata_variables.append(FV.AMB_WD)
    if args.partial_wakes == "gaussian_lookup":
        lookup_model = mbook.partial_wakes["gaussian_lookup"]
        if args.lookup_file is not None:
            lookup_model.lookup_data = args.lookup_file
        lookup_model.min_weight = args.min_weight
        lookup_model.bounds_policy = args.bounds_policy

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        wake_models=[args.wake_model],
        rotor_model=args.rotor,
        partial_wakes=args.partial_wakes,
        wake_frame="rotor_wd",
        mbook=mbook,
        verbosity=max(args.verbosity - 1, 0),
    )

    engine = foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunk_size_states,
        chunk_size_points=args.chunk_size_points,
        verbosity=max(args.verbosity - 1, 0),
    )

    mbook.outputs["flow_plot_xy_states"] = foxes.models.outputs.FlowPlots2D(
        plane="xy",
        reduction="states",
        var=args.var,
        show=True,
        grid_pars=dict(resolution=args.fig_resolution),
        plot_pars=dict(rotor_color="red", figsize=(7, 6)),
        states_isel=[args.fig_state_index],
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
