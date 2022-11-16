import numpy as np
import pandas as pd
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
    parser.add_argument("--we", help="The wind speed error", type=float, default=0.04)
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

    # create lower and upper bound states for a given error in wind
    states_lower = foxes.input.states.SingleStateStates(
        ws=args.ws-(args.we*args.ws), wd=args.wd, ti=args.ti, rho=args.rho
    )

    states_upper = foxes.input.states.SingleStateStates(
        ws=args.ws+(args.we*args.ws), wd=args.wd, ti=args.ti, rho=args.rho
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
    # lower bound algo:
    algo_lower = foxes.algorithms.Downwind(
        mbook,
        farm,
        states=states_lower,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame="rotor_wd",
        partial_wakes_model=args.pwakes,
        chunks=None,
    )

    # upper bound algo:
    algo_upper = foxes.algorithms.Downwind(
        mbook,
        farm,
        states=states_upper,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame="rotor_wd",
        partial_wakes_model=args.pwakes,
        chunks=None,
    )

    # calculate farm results:
    farm_results = algo.calc_farm()
    farm_results_lower = algo_lower.calc_farm()
    farm_results_upper = algo_upper.calc_farm()
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

    # Print farm results data:
    fr = farm_results.to_dataframe()
    print("\nResults summary:\n")
    print(
        fr[[FV.X, FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_TI, FV.TI, FV.AMB_P, FV.P, FV.CT]]
    )
    fr_lower = farm_results_lower.to_dataframe()
    fr_upper = farm_results_upper.to_dataframe()
        
    # print results for power at lower, ws and upper
    print("\nWind speed and power at lower bound:")
    print(fr_lower[[FV.REWS, FV.P]])
    print("\nWind speed and power at upper bound:")
    print(fr_upper[[FV.REWS, FV.P]])

    # create df of error data
    fr_with_bounds = fr[[FV.REWS, FV.P]].copy()
    fr_with_bounds['P_lower'] = fr_lower[[FV.P]]
    fr_with_bounds['P_upper'] = fr_upper[[FV.P]]
    fr_with_bounds['Abs_error'] = (fr_with_bounds['P_upper'] - fr_with_bounds['P_lower'])/2
    fr_with_bounds['Rel_error'] = fr_with_bounds['Abs_error']/fr_with_bounds['P']
    print("\nUpper and lower bounds on power:")
    print(fr_with_bounds)

    # set hours and power factors
    hours_factor = 24 * 365 
    power_factor = 1e-6

    # compute yield and its error
    YLD = fr_with_bounds.P.sum() * hours_factor
    print(f"\nAnnual yield is {YLD*power_factor:.2f} GWh")
    YLD_error_abs = np.sqrt(np.sum((fr_with_bounds['Abs_error'] * hours_factor)**2)) # sum errors in quadrature
    YLD_error_rel = YLD_error_abs/YLD
    print(f"Absolute error of yield is {YLD_error_abs*power_factor:.2f} GWh")
    print(f"Relative error of yield is {YLD_error_rel*100:.1f} %")

    # P75 and P90
    P75 = YLD * (1.0 - (0.675 * YLD_error_rel))
    P90 = YLD * (1.0 - (1.282 * YLD_error_rel))
    print(f"P75 is {P75 * power_factor:.2f} GWh")
    print(f"P90 is {P90* power_factor:.2f} GWh")
    print()

   