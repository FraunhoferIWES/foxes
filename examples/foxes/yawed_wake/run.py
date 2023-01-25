import numpy as np
import argparse
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV

if __name__ == "__main__":

    # define arguments and options:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y", "--yawm", help="YAWM angle of first turbine", default=30.0, type=float
    )
    parser.add_argument(
        "-y2", "--yawm2", help="YAWM angle of second turbine", default=0.0, type=float
    )
    parser.add_argument("--ws", help="The wind speed", type=float, default=9.0)
    parser.add_argument("--wd", help="The wind direction", type=float, default=270.0)
    parser.add_argument("--ti", help="The TI value", type=float, default=0.08)
    parser.add_argument("--rho", help="The air density", type=float, default=1.225)
    parser.add_argument(
        "-dx", "--deltax", help="The turbine distance in x", type=float, default=1500.0
    )
    parser.add_argument(
        "-dy", "--deltay", help="Turbine layout y step", type=float, default=0.0
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
        default=["PorteAgel_linear", "CrespoHernandez_max"],
        nargs="+",
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes model", default="auto"
    )
    parser.add_argument("-f", "--frame", help="The wake frame", default="yawed")
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=["kTI_04"], nargs="+"
    )
    parser.add_argument("-v", "--var", help="The plot variable", default=FV.WS)
    args = parser.parse_args()

    # create model book
    mbook = foxes.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    # set turbines in yaw
    yawm = np.array([[args.yawm, args.yawm2]])
    mbook.turbine_models["set_yawm"] = foxes.models.turbine_models.SetFarmVars()
    mbook.turbine_models["set_yawm"].add_var(FV.YAWM, yawm)

    # create states
    states = foxes.input.states.SingleStateStates(
        ws=args.ws, wd=args.wd, ti=args.ti, rho=args.rho
    )

    # create wind farm
    print("\nCreating wind farm")
    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=np.array([0.0, 0.0]),
        xy_step=np.array([args.deltax, args.deltay]),
        n_turbines=2,
        turbine_models=args.tmodels + ["set_yawm", "yawm2yaw", ttype.name],
    )

    # create algorithm
    algo = foxes.algorithms.Downwind(
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

    # horizontal flow plot
    print("\nHorizontal flow figure output:")
    o = foxes.output.FlowPlots2D(algo, farm_results)
    g = o.gen_states_fig_xy(args.var, resolution=10, xmin=-100, xmax=3000)
    fig = next(g)
    plt.show()
    plt.close(fig)

    # vertical flow plot
    print("\nVertical flow figure output:")
    o = foxes.output.FlowPlots2D(algo, farm_results)
    g = o.gen_states_fig_yz(args.var, resolution=10, x=750, ymin=-200,ymax=200,zmax=300)
    fig = next(g)
    plt.show()
    plt.close(fig)

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
                FV.AMB_REWS,
                FV.REWS,
                FV.AMB_TI,
                FV.TI,
                FV.AMB_P,
                FV.P,
                FV.WD,
                FV.YAW,
                FV.YAWM,
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
    turbine_results[FV.AMB_YLD] = o.calc_turbine_yield(annual=True, ambient=True)
    turbine_results[FV.YLD] = o.calc_turbine_yield(annual=True)
    print("\nResults by turbine:\n")
    print(turbine_results)

    # power results
    P0 = o.calc_mean_farm_power(ambient=True)
    P = o.calc_mean_farm_power()
    print(f"\nFarm power        : {P/1000:.1f} MW")
    print(f"Farm ambient power: {P0/1000:.1f} MW")
    print(f"Farm efficiency   : {o.calc_farm_efficiency()*100:.2f} %")
    print(f"Annual farm yield : {turbine_results[FV.YLD].sum():.2f} GWh.")
