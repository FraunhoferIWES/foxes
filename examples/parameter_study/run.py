import time
import argparse
import numpy as np
from xarray import Dataset

import foxes
import foxes.variables as FV
import foxes.constants as FC

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    states = foxes.input.states.Timeseries(
        data_source="timeseries_3000.csv.gz",
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_grid(
        farm,
        xy_base=np.array([500.0, 500.0]),
        step_vectors=np.array([[500.0, 0], [0, 700.0]]),
        steps=(5, 5),
        turbine_models=args.tmodels + ["NREL5MW"],
    )

    n_pop = 10
    pop_data = Dataset(
        {FV.K: (("i", FC.TURBINE), np.full((n_pop, farm.n_turbines), 0.04))}
    )
    pop_data[FV.K][:, 0] += np.arange(n_pop) / 100
    print("\nInput population data:\n\n", pop_data)
    print(f"\nTurbine 0, {FV.K}: {pop_data[FV.K].values[:, 0]}\n")

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        wake_models=args.wakes,
        rotor_model=args.rotor,
        wake_frame="rotor_wd",
        partial_wakes=args.pwakes,
        population_params=dict(
            data_source=pop_data,
            index_coord="i",
            turbine_coord=FC.TURBINE,
        ),
    )

    with foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
    ):
        time0 = time.time()
        farm_results = algo.calc_farm()
        time1 = time.time()
        print("Calc time =", time1 - time0, "\n")

    pop_results = algo.population_model.farm2pop_results(algo, farm_results)
    print("\nPopulation results:\n\n", pop_results)
    print(f"\nState0, Turbine 0, {FV.K}: {pop_results[FV.K].values[:, 0, 0]}")
