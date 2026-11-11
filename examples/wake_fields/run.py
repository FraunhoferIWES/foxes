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
    parser.add_argument("-o", "--outputs", nargs="+", default=["wake_field"])
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
    if args.nofig:
        for name in args.outputs:
            mbook.outputs[name].show = False
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
    with foxes.Engine.new(
        "numpy",
        chunk_size_states=2,
        chunk_size_points=500,
        verbosity=max(args.verbosity - 1, 0),
    ):
        results = algo.calc_farm(outputs=args.outputs)
    farm_results = (
        results.get("farm_results")
        if isinstance(results, dict)
        else results
        if any(
            isinstance(mbook.outputs[name], foxes.models.outputs.FarmResultsOutput)
            for name in args.outputs
        )
        else None
    )
    if farm_results is not None and args.verbosity > 1:
        print("\nFarm results:\n", farm_results)
    if "wake_field" in args.outputs:
        if args.verbosity > 1:
            print("Wake field written to wake_field.nc")
    else:
        print("Skipping wake-field export: select -o wake_field to write it.")
    if args.verbosity > 1:
        if isinstance(results, dict):
            for name, result in results.items():
                if name != "farm_results":
                    print(f"\n{name}:\n", result)
        elif farm_results is None:
            print(results)


if __name__ == "__main__":
    main()
