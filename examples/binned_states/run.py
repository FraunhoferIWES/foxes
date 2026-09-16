import argparse

import matplotlib.pyplot as plt
import numpy as np

import foxes
import foxes.variables as FV


if __name__ == "__main__":
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
        help="The timeseries input file (path or static)",
        default="timeseries_8000.csv.gz",
    )
    parser.add_argument(
        "-e", "--engine", help="The calculation engine", default="single"
    )
    parser.add_argument(
        "-c",
        "--chunksize_states",
        help="The source-state chunk size",
        default=None,
        type=int,
    )
    parser.add_argument(
        "-nf", "--nofig", help="Do not show the wind-rose canvas", action="store_true"
    )
    args = parser.parse_args()

    source_states = foxes.input.states.Timeseries(
        data_source=args.states,
        output_vars=[FV.WS, FV.WD],
        var2col={FV.WS: "ws", FV.WD: "wd"},
    )

    binned_states = foxes.input.states.BinnedStates(
        source_states,
        bin_vars={
            FV.WS: [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 30.0],
            FV.WD: np.arange(-15.0, 376.0, 30.0),
        },
        support_grid={
            FV.X: np.linspace(99000.0, 106000.0, 9),
            FV.Y: np.linspace(999000.0, 1010000.0, 6),
            FV.H: np.array([90.0]),
        },
        interpolation="linear",
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_from_file(
        farm,
        args.layout,
        col_x="x",
        col_y="y",
        col_H="H",
        turbine_models=["NREL5MW"],
    )

    algo = foxes.algorithms.Downwind(
        farm,
        states=binned_states,
        wake_models=[],
        rotor_model="centre",
        verbosity=1,
    )

    engine = foxes.Engine.new(
        engine_type=args.engine,
        chunk_size_states=args.chunksize_states,
    )

    if args.engine == "process":
        preprocessing_engine = foxes.Engine.new(
            engine_type="single",
            chunk_size_states=args.chunksize_states,
        )
        with preprocessing_engine:
            algo.initialize()
            support_rose_data = binned_states.get_support_wind_rose_data(
                algo._Algorithm__loaded_data
            )

        with engine:
            farm_results = algo.calc_farm(ambient=True)
    else:
        with engine:
            algo.initialize()
            support_rose_data = binned_states.get_support_wind_rose_data(
                algo._Algorithm__loaded_data
            )
            farm_results = algo.calc_farm(ambient=True)

    if not args.nofig:
        fig = binned_states.get_support_wind_roses_figure(
            support_rose_data,
            title="Timeseries wind roses at support points",
        )
        plt.show()
        plt.close(fig)

    print(f"Input states: {source_states.size()}")
    print(f"Histogram states: {binned_states.size()}")
    print("Binned farm results:")
    print(farm_results[[FV.AMB_REWS, FV.AMB_WD, FV.WEIGHT]])
