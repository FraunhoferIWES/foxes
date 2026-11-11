import time
import argparse
import numpy as np
from xarray import Dataset

import foxes
import foxes.variables as FV
import foxes.constants as FC


def main(argv=None):
    """Run this tutorial with an explicit argument list or command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument(
        "-l",
        "--layout",
        help="The wind farm layout file (path or static)",
        default="test_farm_67.csv",
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes models", default=None, nargs="+"
    )
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["Bastankhah2016_linear"],
        nargs="+",
    )
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=[], nargs="+"
    )
    parser.add_argument("-e", "--engine", help="The engine", default=None)
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
        print("Running example: parameter_study")

    mbook = foxes.ModelBook()
    mbook.outputs["farm_results"].fdata_variables.extend([FV.K])

    if args.nofig:
        for name in args.outputs:
            mbook.outputs[name].show = False

    states = foxes.input.states.Timeseries(
        data_source="timeseries_3000.csv.gz",
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_grid(
        farm,
        xy_base=np.array([500.0, 500.0]),
        step_vectors=np.array([[500.0, 0], [0, 700.0]]),
        steps=(args.grid_size, args.grid_size),
        turbine_models=args.tmodels + ["NREL5MW"],
        verbosity=max(args.verbosity - 1, 0),
    )

    k_values = np.linspace(0.01, 0.1, args.population_size)
    n_pop = len(k_values)
    pop_data = np.zeros((n_pop, farm.n_turbines))
    pop_data[:] = k_values[:, None]
    pop_data = Dataset(
        {
            FV.K: (("i", FC.TURBINE), pop_data),
        }
    )
    if args.verbosity > 1:
        print("\nInput population data:\n\n", pop_data)
        print(f"\n{FV.K} values: {pop_data[FV.K].values[:, 0]}\n")

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        mbook=mbook,
        wake_models=args.wakes,
        rotor_model=args.rotor,
        wake_frame="rotor_wd",
        partial_wakes=args.pwakes,
        population_params=dict(
            data_source=pop_data,
            index_coord="i",
            turbine_coord=FC.TURBINE,
            verbosity=max(args.verbosity - 1, 0),
        ),
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
    if isinstance(farm_results, dict):
        farm_results = farm_results.get("farm_results")
    elif not any(
        isinstance(mbook.outputs[name], foxes.models.outputs.FarmResultsOutput)
        for name in args.outputs
    ):
        farm_results = None
    if farm_results is not None and args.verbosity > 1:
        print("\nFarm results:\n", farm_results)
    required = {FV.K, FV.P, FV.WEIGHT}
    if farm_results is None or not required.issubset(farm_results.data_vars):
        print(
            "Skipping farm reporting: selected outputs do not collect the required fields:",
            sorted(required),
        )
        return

    if args.verbosity > 1:
        print("Calc time =", time1 - time0, "\n")

    pop_results = algo.population_model.farm2pop_results(algo, farm_results)
    if args.verbosity > 1:
        print("\nPopulation results:\n\n", pop_results)
        print(f"\n{FV.K} values: {pop_results[FV.K].values[:, 0, 0]}")

    P_mean = (
        (pop_results[FV.P] * pop_results[FV.WEIGHT])
        .sum(dim=FC.STATE)
        .sum(dim=FC.TURBINE)
    )
    if args.verbosity > 1:
        print("\nFarm P_mean:", P_mean.values)


if __name__ == "__main__":
    main()
