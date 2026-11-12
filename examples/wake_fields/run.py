"""Export a scanned wake field through ordinary chunk outputs."""

import argparse
from itertools import product

import numpy as np
import pandas as pd

import foxes
import foxes.variables as FV


def main(argv=None):
    """Run this tutorial with an explicit argument list or command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=["wake_field"],
    )
    parser.add_argument(
        "-nf", "--nofig", action="store_true", help="Do not show figures"
    )
    parser.add_argument("--ws", nargs="+", type=float, default=[6.0, 9.0, 12.0])
    parser.add_argument("--ct", nargs="+", type=float, default=[0.4, 0.8])
    parser.add_argument("--points", type=int, default=31, help="Grid points per axis")
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=2,
        help="0 quiet, 1 summary, 2 reports, 3 library progress",
    )
    args = parser.parse_args(argv)
    if args.verbosity > 0:
        print("Running example: wake_fields")

    # Scan inflow and thrust independently without a separate exporter API.
    cases = np.array(list(product(args.ws, args.ct)))
    states = foxes.input.states.StatesTable(
        pd.DataFrame({FV.WS: cases[:, 0]}),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.WD: 270.0, FV.TI: 0.05, FV.RHO: 1.225},
    )
    mbook = foxes.ModelBook()
    mbook.outputs["farm_results"].fdata_variables.append(FV.CT)
    thrust = foxes.models.turbine_models.SetFarmVars()
    thrust.add_var(FV.CT, cases[:, 1, None])
    mbook.turbine_models["scan_ct"] = thrust
    mbook.outputs["wake_field"] = foxes.models.outputs.SliceData(
        variables=[FV.WS],
        to_file="wake_field.nc",
        grid_pars={
            "xmin": -100,
            "xmax": 1500,
            "ymin": -500,
            "ymax": 500,
            "n_img_points": (args.points, args.points),
        },
    )
    farm = foxes.WindFarm()
    farm.add_turbine(
        foxes.Turbine([0.0, 0.0], turbine_models=["NREL5MW", "scan_ct"]),
        verbosity=max(args.verbosity - 1, 0),
    )
    algo = foxes.algorithms.Downwind(
        farm,
        states,
        mbook=mbook,
        rotor_model="centre",
        wake_models=["Bastankhah2014_linear_k004"],
        verbosity=max(args.verbosity - 1, 0),
    )
    # Apply figure suppression to both calculation stages.
    for name in [*args.results, *args.outputs]:
        output = mbook.outputs[name]
        output.nofig = args.nofig

    with foxes.Engine.new(
        "numpy",
        chunk_size_states=2,
        chunk_size_points=500,
        verbosity=max(args.verbosity - 1, 0),
    ):
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
