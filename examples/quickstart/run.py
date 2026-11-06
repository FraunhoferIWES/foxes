import argparse

import foxes

if __name__ == "__main__":
    # Select chunk outputs by their model-book names.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--outputs", help="The output models", nargs="+", default=["farm_results"]
    )
    parser.add_argument(
        "-nf", "--nofig", help="Do not show figures", action="store_true"
    )
    args = parser.parse_args()

    states = foxes.input.states.Timeseries(
        "timeseries_3000.csv.gz", ["WS", "WD", "TI", "RHO"]
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_from_file(
        farm, "test_farm_67.csv", turbine_models=["NREL5MW"]
    )

    algo = foxes.algorithms.Downwind(farm, states, ["Jensen_linear_k007"])
    farm_results = algo.calc_farm(outputs=args.outputs)

    # Evaluate collected farm data only; writer-only outputs are already complete.
    if isinstance(farm_results, dict):
        farm_results = farm_results.get("farm_results")
    if farm_results is None:
        raise SystemExit(0)

    print(farm_results)
