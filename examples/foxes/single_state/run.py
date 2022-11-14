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
    print("\nResults summary:\n")
    fr = farm_results.to_dataframe()
    fr_lower = farm_results_lower.to_dataframe()
    fr_upper = farm_results_upper.to_dataframe()
    print(
        fr[[FV.X, FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_TI, FV.TI, FV.AMB_P, FV.P, FV.CT]]
    )

    # compute outputs for lower bound
    o = foxes.output.FarmResultsEval(farm_results_lower)
    P0_lower = o.calc_mean_farm_power(ambient=True)
    P_lower = o.calc_mean_farm_power()
    
    # compute outputs for upper bound
    o = foxes.output.FarmResultsEval(farm_results_upper)
    P0_upper = o.calc_mean_farm_power(ambient=True)
    P_upper = o.calc_mean_farm_power()
    
    # compute outputs for ws
    o = foxes.output.FarmResultsEval(farm_results)
    P0 = o.calc_mean_farm_power(ambient=True)
    P = o.calc_mean_farm_power()
    print(f"\nFarm power: {P/1000:.3f} MW, Efficiency = {P/P0*100:.2f} %")
    print(f"Farm power upper: {P_upper/1000:.3f} MW")
    print(f"Farm power lower: {P_lower/1000:.3f} MW")

    # error in power
    std_err_lower = (P-P_lower)/P
    std_err_upper = (P_upper-P)/P
    std_err_mean = (std_err_lower + std_err_upper)/2
    print(f"\nStandard error in wind is {args.we*100} %")
    print(f"Standard error in power per turbine is {std_err_mean*100:.1f} %")

    # yield calculations 
    Y_turbine = o.calc_turbine_yield(hours=24*365, power_factor=0.000001, power_uncert=std_err_mean, ambient=False) # results will be in GWh per year
    print(f"\n Yield per turbine [GWh]:")
    print(Y_turbine)
    farm_yield, P75, P90 = o.calc_farm_yield(hours=24*365, power_factor=0.000001, power_uncert=std_err_mean, ambient=False) # results will be in GWh per year
    farm_yield_AMB, P75_AMB, P90_AMB = o.calc_farm_yield(hours=24*365, power_factor=0.000001, power_uncert=std_err_mean, ambient=True)

    print(f"\nFarm yield: {farm_yield:.1f} GWh")
    print(f"Farm wake losses: {farm_yield_AMB - farm_yield:.1f} GWh")
    print(f"Farm P75: {P75:.1f} GWh")
    print(f"Farm P90: {P90:.1f} GWh")