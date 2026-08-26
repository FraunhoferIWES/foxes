import argparse
import foxes
import foxes.variables as FV
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("-v", "--var", help="Flow plot variable", default=FV.WS)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    states = foxes.input.states.Timeseries(
        args.timeseries,
        [FV.WS, FV.WD, FV.TI, FV.RHO],
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_from_file(
        farm,
        args.layout,
        turbine_models=[args.turbine_type],
    )

    mbook = foxes.models.ModelBook()
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
        verbosity=1,
    )

    engine = foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunk_size_states,
        chunk_size_points=args.chunk_size_points,
    )

    with engine:
        farm_results = algo.calc_farm()

        plot_data = None
        if not args.nofig:
            out = foxes.output.FlowPlots2D(algo, farm_results)
            plot_data = out.get_states_data_xy(
                args.var,
                resolution=args.fig_resolution,
                states_isel=[args.fig_state_index],
            )

    print("\nUsing partial_wakes:", args.partial_wakes)
    print("\nFarm results:\n", farm_results)

    if not args.nofig and plot_data is not None:
        fig_gen = out.gen_states_fig_xy(plot_data, rotor_color="red", figsize=(7, 6))
        fig = next(fig_gen)
        plt.show()


if __name__ == "__main__":
    main()
