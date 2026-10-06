import argparse
from math import isqrt
import time

import matplotlib.pyplot as plt
import numpy as np

import foxes
import foxes.variables as FV
from foxes.config import config
from foxes.utils import from_lonlat, get_utm_zone


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ws",
        "--wind-speed-file",
        default="pecd_wind_speed.csv",
        help="Path to the wind-speed input file",
    )
    parser.add_argument(
        "-wd",
        "--wind-direction-file",
        default="pecd_wind_direction.csv",
        help="Path to the wind-direction input file",
    )
    parser.add_argument(
        "--extrapolation-heights",
        type=float,
        nargs="+",
        help="Additional heights in metres for ABL-log wind-speed extrapolation",
    )
    parser.add_argument(
        "--profile-z0",
        type=float,
        help="Roughness length in metres; required for height extrapolation",
    )
    parser.add_argument(
        "--profile-mol",
        type=float,
        default=np.nan,
        help="Monin-Obukhov length in metres (default: neutral profile)",
    )
    parser.add_argument(
        "-t",
        "--turbine-file",
        default="NREL-5MW-D126-H90.csv",
        help="The P-ct-curve CSV file (path or static data name)",
    )
    parser.add_argument(
        "-nt",
        "--n-turbines",
        default=9,
        type=int,
        help="Number of turbines; turbines are arranged on a square grid",
    )
    parser.add_argument("-e", "--engine", default="single", help="FOXES engine type")
    parser.add_argument(
        "-n",
        "--n-cpus",
        default=1,
        type=int,
        help="Number of processes for engines that support multiprocessing",
    )
    parser.add_argument(
        "-c",
        "--chunksize-states",
        default=64,
        type=int,
        help="Number of states per calculation chunk",
    )
    parser.add_argument(
        "--slice-resolution",
        default=2000.0,
        type=float,
        help="Resolution in metres for state-resolved flow slices",
    )
    parser.add_argument(
        "--flow-resolution",
        default=1000.0,
        type=float,
        help="Resolution in metres for the mean flow plot",
    )
    parser.add_argument("-nf", "--nofig", action="store_true")
    args = parser.parse_args()
    n_side = isqrt(args.n_turbines)
    if args.n_turbines < 1 or n_side**2 != args.n_turbines:
        parser.error("--n-turbines must be a positive square number")
    return args


def main() -> None:
    args = _parse_args()
    slice_heights = sorted({100.0, *(args.extrapolation_heights or [])})
    states = foxes.input.states.PECDStates(
        args.wind_speed_file,
        args.wind_direction_file,
        height=100.0,
        extrapolation_heights=args.extrapolation_heights,
        profile_z0=args.profile_z0,
        profile_mol=args.profile_mol,
        output_vars=[FV.WS, FV.WD],
        fixed_vars={FV.RHO: 1.225},
    )
    data = states.data_source
    centre_lonlat = np.array(
        [[data["longitude"].mean().item(), data["latitude"].mean().item()]]
    )
    config.set_utm_zone(*get_utm_zone(centre_lonlat))
    farm_centre = from_lonlat(centre_lonlat)[0]

    mbook = foxes.models.ModelBook()
    turbine_type = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[turbine_type.name] = turbine_type

    n_side = isqrt(args.n_turbines)
    turbine_spacing = 500.0
    offset = turbine_spacing * (n_side - 1) / 2
    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_grid(
        farm,
        xy_base=farm_centre - np.array([offset, offset]),
        step_vectors=np.array(
            [[turbine_spacing, 0.0], [0.0, turbine_spacing]], dtype=np.float64
        ),
        steps=(n_side, n_side),
        H=100.0,
        turbine_models=[turbine_type.name],
        verbosity=0,
    )

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        wake_models=["Jensen_linear_k007"],
        rotor_model="centre",
        wake_frame="rotor_wd",
        partial_wakes="centre",
        mbook=mbook,
        verbosity=2,
    )
    engine = foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
    )

    with engine:
        time0 = time.time()
        farm_results = algo.calc_farm()
        time1 = time.time()

    print("\nCalc time =", time1 - time0, "\n")
    print(farm_results)
    result_table = farm_results.to_dataframe()
    print(result_table[[FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]])

    slices = foxes.output.SlicesData(algo, farm_results)
    with engine:
        slice_data = slices.get_states_data_xy(
            z_list=slice_heights,
            variables=[FV.WS],
            resolution=args.slice_resolution,
            verbosity=1,
        )
    print(slice_data)

    if not args.nofig:
        flow_plots = foxes.output.FlowPlots2D(algo, farm_results)
        with engine:
            mean_data = flow_plots.get_mean_data_xy(
                FV.WS, resolution=args.flow_resolution
            )
        flow_plots.get_mean_fig_xy(mean_data)
        plt.show()


if __name__ == "__main__":
    main()
