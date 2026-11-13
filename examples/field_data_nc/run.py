import argparse
import numpy as np

import foxes
import foxes.variables as FV


def main(argv=None):
    """Run this tutorial with an explicit argument list or command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file_pattern",
        help="The search pattern for input *.nc files",
        default="data/data_*.nc",
    )
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes models", default="centre", nargs="+"
    )
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["Jensen_linear_k007"],
        nargs="+",
    )
    parser.add_argument(
        "-wf", "--wake_frame", help="The wake frame choice", default="rotor_wd"
    )
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=[], nargs="+"
    )
    parser.add_argument(
        "-nt", "--n_turbines", help="The number of turbines", default=9, type=int
    )
    parser.add_argument("-e", "--engine", help="The engine", default="ProcessEngine")
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
        "-lm",
        "--load_mode",
        help="Do load mode",
        default="preload",
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
        default=["slices", "flow_plot_xy_mean"],
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
        print("Running example: field_data_nc")

    states = foxes.input.states.FieldData(
        args.file_pattern,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        load_mode=args.load_mode,
        bounds_extra_space=None,
        height_bounds=None,
    )

    mbook = foxes.models.ModelBook()
    mbook.outputs["farm_results"].fdata_variables.append(FV.AMB_WD)
    mbook.outputs["slices"] = foxes.models.outputs.SlicesData(
        plane="xy",
        reduction="states",
        positions=[90, 100],
        variables=[FV.WS],
        grid_pars=dict(resolution=50),
        result_name="slices",
    )

    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    farm = foxes.WindFarm()
    N = int(args.n_turbines**0.5)
    foxes.input.farm_layout.add_grid(
        farm,
        xy_base=np.array([500.0, 500.0]),
        step_vectors=np.array([[500.0, 0], [0, 500.0]]),
        steps=(N, N),
        turbine_models=args.tmodels + [ttype.name],
        verbosity=max(args.verbosity - 1, 0),
    )

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        wake_models=args.wakes,
        rotor_model=args.rotor,
        wake_frame=args.wake_frame,
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

    mbook.outputs["flow_plot_xy_mean"] = foxes.models.outputs.FlowPlots2D(
        plane="xy",
        reduction="mean",
        var=FV.WS,
        show=True,
        grid_pars=dict(resolution=10),
        plot_pars=dict(),
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
