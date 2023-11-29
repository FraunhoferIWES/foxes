import numpy as np
import argparse
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV

if __name__ == "__main__":
    # define arguments and options:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nt", "--n_t", help="The number of turbines", type=int, default=10
    )
    parser.add_argument(
        "-l",
        "--layout",
        help="The wind farm layout file (path or static)",
        default=None,
    )
    parser.add_argument("--ws", help="The wind speed", type=float, default=9.0)
    parser.add_argument("--wd", help="The wind direction", type=float, default=270.0)
    parser.add_argument("--ti", help="The TI value", type=float, default=0.08)
    parser.add_argument("--rho", help="The air density", type=float, default=1.225)
    parser.add_argument(
        "-dx", "--deltax", help="The turbine distance in x", type=float, default=0.0
    )
    parser.add_argument(
        "-dy", "--deltay", help="Turbine layout y step", type=float, default=200.0
    )
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=[], nargs="+"
    )
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["RHB_linear", "Bastankhah_linear_k002"],
        nargs="+",
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes model", default="auto"
    )
    parser.add_argument("-f", "--frame", help="The wake frame", default="rotor_wd")
    parser.add_argument("-v", "--var", help="The plot variable", default=FV.WS)
    parser.add_argument(
        "-nit",
        "--not_iterative",
        help="Don't use the iterative algorithm",
        action="store_true",
    )
    args = parser.parse_args()

    # create model book
    mbook = foxes.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    # create states
    states = foxes.input.states.SingleStateStates(
        ws=args.ws, wd=args.wd, ti=args.ti, rho=args.rho
    )

    # create wind farm
    print("\nCreating wind farm")
    farm = foxes.WindFarm()
    if args.layout is None:
        foxes.input.farm_layout.add_row(
            farm=farm,
            xy_base=np.array([0.0, 0.0]),
            xy_step=np.array([args.deltax, args.deltay]),
            n_turbines=args.n_t,
            turbine_models=args.tmodels + [ttype.name],
        )
    else:
        foxes.input.farm_layout.add_from_file(
            farm, args.layout, turbine_models=args.tmodels + [ttype.name]
        )

    # create algorithm
    Algo = (
        foxes.algorithms.Downwind if args.not_iterative else foxes.algorithms.Iterative
    )
    algo = Algo(
        mbook,
        farm,
        states=states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame=args.frame,
        partial_wakes_model=args.pwakes,
        chunks=None,
    )

    # calculate farm results
    farm_results = algo.calc_farm()
    print("\nResults data:\n", farm_results)

    # add capacity and efficiency to farm results
    o = foxes.output.FarmResultsEval(farm_results)
    o.add_capacity(algo)
    o.add_capacity(algo, ambient=True)
    o.add_efficiency()

    # state-turbine results
    farm_df = farm_results.to_dataframe()
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
                FV.CT,
                FV.EFF,
            ]
        ]
    )
    print()

    # results by turbine
    turbine_results = o.reduce_states(
        {
            FV.AMB_P: "mean",
            FV.P: "mean",
            FV.AMB_CAP: "mean",
            FV.CAP: "mean",
            FV.EFF: "mean",
        }
    )
    turbine_results[FV.AMB_YLD] = o.calc_turbine_yield(
        algo=algo, annual=True, ambient=True
    )
    turbine_results[FV.YLD] = o.calc_turbine_yield(algo=algo, annual=True)
    print("\nResults by turbine:\n")
    print(turbine_results)

    # power results
    P0 = o.calc_mean_farm_power(ambient=True)
    P = o.calc_mean_farm_power()
    print(f"\nFarm power        : {P/1000:.1f} MW")
    print(f"Farm ambient power: {P0/1000:.1f} MW")
    print(f"Farm efficiency   : {o.calc_farm_efficiency()*100:.2f} %")
    print(f"Annual farm yield : {turbine_results[FV.YLD].sum():.2f} GWh.")

    # horizontal flow plot
    o = foxes.output.FlowPlots2D(algo, farm_results)
    g = o.gen_states_fig_xy(
        args.var,
        figsize=(4, 8),
        resolution=2,
        xmin=-300,
        xmax=args.n_t * args.deltax + 500,
        yspace=200.0,
    )
    fig = next(g)
    plt.show()
    plt.close(fig)

    # center line plot:
    H = mbook.turbine_types[ttype.name].H
    n_points = 8000
    points = np.zeros((1, n_points, 3))
    points[:, :, 0] = np.linspace(-500.0, 2000.0, n_points)[None, :]
    points[:, :, 2] = H
    point_results = algo.calc_points(farm_results, points)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(points[0, :, 0], point_results[FV.WS][0, :])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Wind speed [m/s]")
    plt.show()
    plt.close(fig)

<<<<<<< HEAD
    # horizontal flow plot with wind direction
    o = foxes.output.FlowPlots2D(algo, farm_results)
    g = o.gen_states_fig_xy(args.var, resolution=2, quiver_pars=dict(angles="xy", scale_units="xy", scale=0.35),
        quiver_n=4,xspace=120.0, yspace=120.0)
    fig = next(g)
    plt.savefig("RHB_xy.png")
    plt.show()
    plt.close(fig)

    # vertical flow plot with wind direction
    o = foxes.output.FlowPlots2D(algo, farm_results)
    g = o.gen_states_fig_xz(args.var, resolution=2, quiver_pars=dict(angles="xy", scale_units="xy", scale=1.0),
        quiver_n=4,xspace=100.0, zspace=120.0)
    fig = next(g)
    plt.savefig("RHB_xz.png")
    plt.show()
    plt.close(fig)

=======
    # front line plot:
    n_points = 8000
    points = np.zeros((1, n_points, 3))
    points[:, :, 0] = -200
    points[:, :, 1] = np.linspace(-500.0, args.n_t * args.deltay + 500, n_points)[
        None, :
    ]
    points[:, :, 2] = H
    point_results = algo.calc_points(farm_results, points)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(point_results[FV.WS][0, :], points[0, :, 1])
    ax.set_ylabel("y [m]")
    ax.set_xlabel("Wind speed [m/s]")
    plt.show()
    plt.close(fig)
>>>>>>> refs/remotes/origin/dev_blockage
