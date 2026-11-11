import numpy as np
import time
import argparse
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV


def main(argv=None):
    """Run this tutorial with an explicit argument list or command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_s", help="The number of states", type=int, default=30)
    parser.add_argument("-n_t", help="The number of turbines", type=int, default=5)
    parser.add_argument("--n_p", help="The number of turbines", type=int, default=2000)
    parser.add_argument("--ws0", help="The lowest wind speed", type=float, default=3.0)
    parser.add_argument(
        "--ws1", help="The highest wind speed", type=float, default=30.0
    )
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["Bastankhah2014_linear"],
        nargs="+",
    )
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=["kTI_04"], nargs="+"
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes models", default=None, nargs="+"
    )
    parser.add_argument(
        "-cl", "--calc_cline", help="Calculate centreline", action="store_true"
    )
    parser.add_argument(
        "-cm", "--calc_mean", help="Calculate states mean", action="store_true"
    )
    parser.add_argument(
        "-it", "--iterative", help="Use iterative algorithm", action="store_true"
    )
    parser.add_argument("-e", "--engine", help="The engine", default="process")
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
        print("Running example: scan_row")

    n_s = args.n_s
    n_t = args.n_t
    n_p = args.n_p
    p0 = np.array([0.0, 0.0])
    stp = np.array([500.0, 0.0])

    mbook = foxes.models.ModelBook()
    mbook.outputs["farm_results"].fdata_variables.extend([FV.AMB_WD, FV.EFF])

    if args.nofig:
        for name in args.outputs:
            output = mbook.outputs[name]
            if hasattr(output, "show"):
                output.show = False
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype
    D = ttype.D
    H = ttype.H

    states = foxes.input.states.ScanStates(
        {
            FV.WS: np.linspace(args.ws0, args.ws1, n_s),
            FV.WD: [270],
            FV.TI: [0.08],
            FV.RHO: [1.225],
        }
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=p0,
        xy_step=stp,
        n_turbines=n_t,
        turbine_models=args.tmodels + [ttype.name],
        verbosity=max(args.verbosity - 1, 0),
    )

    Algo = foxes.algorithms.Iterative if args.iterative else foxes.algorithms.Downwind
    algo = Algo(
        farm,
        states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame="rotor_wd",
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
        # Request per-state metrics through output selection.
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
    required = {
        FV.AMB_P,
        FV.AMB_REWS,
        FV.AMB_TI,
        FV.EFF,
        FV.P,
        FV.REWS,
        FV.TI,
        FV.WD,
        FV.X,
    }
    if farm_results is None or not required.issubset(farm_results.data_vars):
        print(
            "Skipping farm reporting: selected outputs do not collect the required fields:",
            sorted(required),
        )
        return

    if args.verbosity > 1:
        print("\nCalc time =", time1 - time0, "\n")

    # state-turbine results
    farm_df = farm_results.to_dataframe()
    if args.verbosity > 1:
        print("\nFarm results data:\n")
        print(
            farm_df[
                [
                    FV.X,
                    FV.WD,
                    FV.AMB_REWS,
                    FV.REWS,
                    FV.AMB_TI,
                    FV.TI,
                    FV.AMB_P,
                    FV.P,
                    FV.EFF,
                ]
            ]
        )
        print()

        print()

    if not args.nofig and args.calc_cline:
        points = np.zeros((n_s, n_p, 3))
        points[:, :, 0] = np.linspace(p0[0], p0[0] + n_t * stp[0] + 10 * D, n_p)[
            None, :
        ]
        points[:, :, 1] = p0[1]
        points[:, :, 2] = H
        if args.verbosity > 1:
            print("\nPOINTS:\n", points[0])

        with engine:
            time0 = time.time()
            point_results = algo.calc_points(farm_results, points)
            time1 = time.time()

        if args.verbosity > 1:
            print("\nCalc time =", time1 - time0, "\n")

            print(point_results)

        fig, ax = plt.subplots()
        for s in range(points.shape[0]):
            ax.plot(points[s, :, 0], point_results[FV.WS][s, :])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("Wind speed [m/s]")
        ax.set_title("Centreline wind speed")
        plt.show()
        plt.close(fig)

    if not args.nofig and args.calc_mean:
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
