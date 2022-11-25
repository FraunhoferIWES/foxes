import numpy as np
import argparse
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV

if __name__ == "__main__":

    # define arguments and options:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nt", "--n_t", help="The number of turbines", type=int, default=5
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
        "-d", "--dist_x", help="The turbine distance in x", type=float, default=500.0
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
        default=["Bastankhah_quadratic", "CrespoHernandez_max"],
        nargs="+",
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes model", default="rotor_points"
    )
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=["kTI_02"], nargs="+"
    )
    parser.add_argument(
        "-dy", "--deltay", help="Turbine layout y step", type=float, default=0.0
    )
    parser.add_argument("-v", "--var", help="The plot variable", default=FV.WS)
    args = parser.parse_args()

    # create model book:
    mbook = foxes.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    # create states:
    states = foxes.input.states.SingleStateStates(
        ws=args.ws, wd=args.wd, ti=args.ti, rho=args.rho
    )

    # create wind farm:
    print("\nCreating wind farm")
    farm = foxes.WindFarm()
    if args.layout is None:
        foxes.input.farm_layout.add_row(
            farm=farm,
            xy_base=np.array([0.0, 0.0]),
            xy_step=np.array([args.dist_x, args.deltay]),
            n_turbines=args.n_t,
            turbine_models=args.tmodels + [ttype.name],
        )
    else:
        foxes.input.farm_layout.add_from_file(
            farm, args.layout, turbine_models=args.tmodels + [ttype.name]
        )

    # create algorithm:
    algo = foxes.algorithms.Downwind(
        mbook,
        farm,
        states=states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame="rotor_wd",
        partial_wakes_model=args.pwakes,
        chunks=None,
    )
    # calculate farm results:
    farm_results = algo.calc_farm()
    print("\nResults data:\n", farm_results)

    # Horizontal flow plot:
    print("\nHorizontal flow figure output:")
    o = foxes.output.FlowPlots2D(algo, farm_results)
    g = o.gen_states_fig_horizontal(args.var, resolution=10)
    fig = next(g)
    plt.show()
    plt.close(fig)

    # Vertical flow plot:
    print("\nVertical flow figure output:")
    o = foxes.output.FlowPlots2D(algo, farm_results)
    g = o.gen_states_fig_vertical(
        args.var, resolution=10, x_direction=np.mod(args.wd + 180, 360.0)
    )
    fig = next(g)
    plt.show()

  # add capacity to farm results
    o = foxes.output.FarmResultsEval(farm_results)
    P_nominal = [t.P_nominal for t in algo.farm_controller.turbine_types] 
    o.calc_capacity(P_nom=P_nominal)
    o.calc_capacity(P_nom=P_nominal, ambient=True)
    
    # add efficiency to farm results
    o.calc_efficiency()

    farm_df = farm_results.to_dataframe()
    print("\nFarm results data:")
    print(farm_df[[FV.X, FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_TI, FV.TI,
    FV.P, FV.AMB_P, FV.CT, FV.EFF, FV.CAP, FV.AMB_CAP]])
    print()

    # results by turbine 
    turbine_results = o.reduce_states({FV.P: "mean",
    FV.AMB_P: "mean",
    FV.CAP: "mean",
    FV.AMB_CAP: "mean",
    FV.EFF: "mean"
    })
    print("\nResults by turbine")
    print(turbine_results)

    # yield results
    turbine_yield = turbine_results['P'] *24*365 *1e-6
    print(f"\nAnnual yield [GWh] per turbine is:")
    print(turbine_yield)
    print(f"\nAnnual farm yield is {turbine_yield.sum():.2f} [GWh].")
    
    # power results
    P0 = o.calc_mean_farm_power(ambient=True)
    P = o.calc_mean_farm_power()
    print(f"\nFarm power: {P/1000:.1f} MW")
    print(f"Farm ambient power: {P0/1000:.1f} MW")