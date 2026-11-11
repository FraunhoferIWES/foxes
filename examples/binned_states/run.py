import argparse

import matplotlib.pyplot as plt
import numpy as np

import foxes
import foxes.variables as FV


def main(argv=None):
    """Run this tutorial with an explicit argument list or command-line options."""
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
    parser.add_argument("-e", "--engine", help="The calculation engine", default=None)
    parser.add_argument(
        "-c",
        "--chunksize_states",
        help="The source-state chunk size",
        default=None,
        type=int,
    )
    parser.add_argument(
        "-n", "--n_cpus", help="The number of cpus", default=None, type=int
    )
    parser.add_argument(
        "-d",
        "--wd_bins",
        help="The number of wind-direction bins",
        default=36,
        type=int,
    )
    parser.add_argument(
        "-b",
        "--ws_bins",
        help="The number of wind-speed bins",
        default=10,
        type=int,
    )
    parser.add_argument(
        "-nf", "--nofig", help="Do not show the wind-rose canvas", action="store_true"
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p",
        "--pwakes",
        help="The partial wakes models",
        default="centre",
        nargs="+",
    )
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["Bastankhah2014"],
        nargs="+",
    )
    parser.add_argument("-f", "--frame", help="The wake frame", default="rotor_wd")
    parser.add_argument(
        "-F",
        "--compare-full-timeseries",
        help="Also run the full timeseries calculation and compare weighted means",
        action="store_true",
    )
    parser.add_argument(
        "--write-nc",
        help="Write the reduced binned data to this NetCDF file",
        default=None,
    )
    parser.add_argument(
        "--read-nc",
        help="Read reduced binned data from this NetCDF file instead of reducing the source states",
        default=None,
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
        print("Running example: binned_states")

    mbook = foxes.ModelBook()
    mbook.outputs["farm_results"].fdata_variables.extend([FV.AMB_WD])

    if args.nofig:
        for name in args.outputs:
            mbook.outputs[name].show = False

    source_states = foxes.input.states.Timeseries(
        data_source=args.states,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd"},
        fixed_vars={FV.RHO: 1.225, FV.TI: 0.07},
    )

    source_binned_data = foxes.input.states.BinnedFieldData(
        source_states,
        bin_vars={
            FV.WS: np.linspace(0.0, 30.0, args.ws_bins + 1),
            FV.WD: args.wd_bins,
        },
        support_grid={
            FV.X: np.linspace(99000.0, 106000.0, 9),
            FV.Y: np.linspace(999000.0, 1010000.0, 6),
            FV.H: np.array([90.0]),
        },
        interpolation="linear",
        output_file=args.write_nc,
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_from_file(
        farm,
        args.layout,
        col_x="x",
        col_y="y",
        col_H="H",
        turbine_models=["NREL5MW"],
        verbosity=max(args.verbosity - 1, 0),
    )

    source_algo = None
    if args.write_nc is not None and args.read_nc is not None:
        source_algo = foxes.algorithms.Downwind(
            farm,
            states=source_binned_data,
            wake_models=args.wakes,
            wake_frame=args.frame,
            partial_wakes=args.pwakes,
            rotor_model=args.rotor,
            verbosity=max(args.verbosity - 1, 0),
        )

    engine = foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        verbosity=max(args.verbosity - 1, 0),
    )

    full_algo = None
    if args.compare_full_timeseries:
        full_states = foxes.input.states.Timeseries(
            data_source=args.states,
            output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
            var2col={FV.WS: "ws", FV.WD: "wd"},
            fixed_vars={FV.RHO: 1.225, FV.TI: 0.07},
        )
        full_algo = foxes.algorithms.Downwind(
            farm,
            states=full_states,
            mbook=mbook,
            wake_models=args.wakes,
            wake_frame=args.frame,
            partial_wakes=args.pwakes,
            rotor_model=args.rotor,
            verbosity=max(args.verbosity - 1, 0),
        )

    with engine:
        support_rose_data = None
        if source_algo is not None:
            source_algo.initialize()
            support_rose_data = source_binned_data.get_support_wind_rose_data(
                source_algo._Algorithm__loaded_data
            )

        binned_data = (
            foxes.input.states.read_binned_data(args.read_nc)
            if args.read_nc is not None
            else source_binned_data
        )
        algo = foxes.algorithms.Downwind(
            farm,
            states=binned_data,
            mbook=mbook,
            wake_models=args.wakes,
            wake_frame=args.frame,
            partial_wakes=args.pwakes,
            rotor_model=args.rotor,
            verbosity=max(args.verbosity - 1, 0),
        )
        if args.read_nc is None:
            algo.initialize()
            support_rose_data = source_binned_data.get_support_wind_rose_data(
                algo._Algorithm__loaded_data
            )
        farm_results = algo.calc_farm(outputs=args.outputs)
        if full_algo is not None:
            if args.verbosity > 1:
                print("\nRunning full timeseries calculation for comparison")
            full_farm_results = full_algo.calc_farm(outputs=args.outputs)

    # Evaluate collected farm data only; complete both runs for writer-only outputs.
    if isinstance(farm_results, dict):
        farm_results = farm_results.get("farm_results")
    elif not any(
        isinstance(mbook.outputs[name], foxes.models.outputs.FarmResultsOutput)
        for name in args.outputs
    ):
        farm_results = None
    if farm_results is not None and args.verbosity > 1:
        print("\nFarm results:\n", farm_results)
    if full_algo is not None:
        if isinstance(full_farm_results, dict):
            full_farm_results = full_farm_results.get("farm_results")
        elif not any(
            isinstance(mbook.outputs[name], foxes.models.outputs.FarmResultsOutput)
            for name in args.outputs
        ):
            full_farm_results = None
        if full_farm_results is not None and args.verbosity > 1:
            print("\nFull timeseries farm results:\n", full_farm_results)
    required = {FV.AMB_REWS, FV.AMB_WD, FV.REWS, FV.WD, FV.WEIGHT}
    if farm_results is None or not required.issubset(farm_results.data_vars):
        print(
            "Skipping farm reporting: selected outputs do not collect the required fields:",
            sorted(required),
        )
        return
    if not args.nofig and support_rose_data is not None:
        fig = source_binned_data.get_support_wind_roses_figure(
            support_rose_data,
            title="Timeseries wind roses at support points",
        )
        plt.show()
        plt.close(fig)

    if args.verbosity > 1:
        print(f"Input states: {source_states.size()}")
        print(f"Histogram states: {binned_data.size()}")
    if args.write_nc is not None:
        if args.verbosity > 1:
            print(f"Written binned data file: {args.write_nc}")
    if args.read_nc is not None:
        if args.verbosity > 1:
            print(f"Read binned data file: {args.read_nc}")

    if args.compare_full_timeseries:
        variables = [FV.AMB_REWS, FV.REWS, FV.AMB_WD, FV.WD]
        binned_weight = farm_results[FV.WEIGHT].to_numpy()
        full_weight = full_farm_results[FV.WEIGHT].to_numpy()
        binned_mean = {}
        full_mean = {}
        for var in variables:
            for results, weight, mean in [
                (farm_results, binned_weight, binned_mean),
                (full_farm_results, full_weight, full_mean),
            ]:
                data = results[var].to_numpy()
                if weight.ndim == 1:
                    weight = weight[:, None]
                valid = np.isfinite(data) & np.isfinite(weight) & (weight > 0.0)
                w = np.where(valid, weight, 0.0)
                if var in {FV.AMB_WD, FV.WD}:
                    angles = np.deg2rad(np.where(valid, data, 0.0))
                    sine = np.sum(w * np.sin(angles), axis=0)
                    cosine = np.sum(w * np.cos(angles), axis=0)
                    mean[var] = np.mod(np.rad2deg(np.arctan2(sine, cosine)), 360.0)
                else:
                    numerator = np.sum(np.where(valid, weight * data, 0.0), axis=0)
                    denominator = np.sum(w, axis=0)
                    mean[var] = np.divide(
                        numerator,
                        denominator,
                        out=np.full_like(numerator, np.nan),
                        where=denominator > 0.0,
                    )

        if args.verbosity > 1:
            print("\nComparison, binned minus full timeseries:")
        for var in variables:
            binned_data = binned_mean[var]
            full_data = full_mean[var]
            if var in {FV.AMB_WD, FV.WD}:
                delta = np.mod(binned_data - full_data + 180.0, 360.0) - 180.0
                diff_label = "angular difference"
            else:
                delta = binned_data - full_data
                diff_label = "difference"
            if args.verbosity > 1:
                print(
                    f"  {var}: max absolute {diff_label} = "
                    f"{np.nanmax(np.abs(delta)):.3f}, mean absolute {diff_label} = "
                    f"{np.nanmean(np.abs(delta)):.3f}"
                )


if __name__ == "__main__":
    main()
