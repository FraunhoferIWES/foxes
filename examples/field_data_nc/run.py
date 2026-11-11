import time
import argparse
import matplotlib.pyplot as plt
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
        "-n", "--n_cpus", help="The number of cpus", default=8, type=int
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
        "-o",
        "--outputs",
        help="The output models",
        nargs="+",
        default=["farm_results", "slices"],
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

    if args.nofig:
        for name in args.outputs:
            output = mbook.outputs[name]
            if hasattr(output, "show"):
                output.show = False
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

    with engine:
        time0 = time.time()
        farm_results = algo.calc_farm(outputs=args.outputs)
        time1 = time.time()

    # Evaluate collected farm data only; writer-only outputs are already complete.
    slices = (
        farm_results.get("slices")
        if isinstance(farm_results, dict)
        else farm_results
        if args.outputs == ["slices"]
        else None
    )
    if slices is not None:
        if args.verbosity > 1:
            print(slices)
    else:
        print("Skipping numerical slices: select -o slices to collect them.")
    if isinstance(farm_results, dict):
        farm_results = farm_results.get("farm_results")
    elif not any(
        isinstance(mbook.outputs[name], foxes.models.outputs.FarmResultsOutput)
        for name in args.outputs
    ):
        farm_results = None
    if farm_results is not None and args.verbosity > 1:
        print("\nFarm results:\n", farm_results)
    required = {FV.AMB_P, FV.AMB_REWS, FV.P, FV.REWS, FV.WD}
    if farm_results is None or not required.issubset(farm_results.data_vars):
        print(
            "Skipping farm reporting: selected outputs do not collect the required fields:",
            sorted(required),
        )
        return

    if args.verbosity > 1:
        print("\nCalc time =", time1 - time0, "\n")

    fr = farm_results.to_dataframe()
    if args.verbosity > 1:
        print(fr[[FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]])

    if not args.nofig:
        with engine:
            fig = algo.calc_outputs(
                outputs=foxes.models.outputs.FlowPlots2D(
                    plane="xy",
                    reduction="mean",
                    plot=True,
                    data_format="numpy",
                    var=FV.WS,
                    grid_pars=dict(resolution=10),
                ),
                farm_results=farm_results,
            )
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
