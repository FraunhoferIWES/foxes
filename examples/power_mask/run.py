import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nt", "--n_t", help="The number of turbines", type=int, default=5
    )
    parser.add_argument(
        "-s",
        "--states",
        help="The timeseries input file (path or static)",
        default="states.csv",
    )
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument(
        "-dx", "--deltax", help="The turbine distance in x", type=float, default=500.0
    )
    parser.add_argument(
        "-dy", "--deltay", help="Turbine layout y step", type=float, default=0.0
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes models", default=None, nargs="+"
    )
    parser.add_argument(
        "-f", "--pmax_file", help="The max_P csv file", default="power_mask.csv"
    )
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["Jensen_linear_k007"],
        nargs="+",
    )
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=[], nargs="+"
    )
    parser.add_argument(
        "-sl",
        "--show_layout",
        help="Flag for showing layout figure",
        action="store_true",
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
    args = parser.parse_args()

    print("\nReading file", args.pmax_file)
    cols = [f"Pmax_{i}" for i in range(args.n_t)]
    Pmax_data = pd.read_csv(args.pmax_file, usecols=cols).to_numpy()

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype
    mbook.turbine_models["set_Pmax"] = foxes.models.turbine_models.SetFarmVars()
    mbook.turbine_models["set_Pmax"].add_var(FV.MAX_P, Pmax_data)
    models = args.tmodels + ["set_Pmax", ttype.name, "PMask"]

    with foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
    ):
        if not args.nofig:
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            o = foxes.output.TurbineTypeCurves(mbook)
            o.plot_curves(ttype.name, [FV.P, FV.CT], axs=axs, P_max=3000.0)
            plt.show()
            plt.close(fig)

            """
            # TODO: ct needs fix
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            o = foxes.output.TurbineTypeCurves(mbook)
            o.plot_curves(ttype.name, [FV.P, FV.CT], axs=axs, P_max=6000.0)
            plt.show()
            plt.close(fig)
            """

        states = foxes.input.states.StatesTable(
            data_source=args.states,
            output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
            fixed_vars={FV.WD: 270.0, FV.TI: 0.05, FV.RHO: 1.225},
        )

        farm = foxes.WindFarm()
        foxes.input.farm_layout.add_row(
            farm=farm,
            xy_base=[0.0, 0.0],
            xy_step=[args.deltax, args.deltay],
            n_turbines=args.n_t,
            turbine_models=models,
        )

        if not args.nofig and args.show_layout:
            ax = foxes.output.FarmLayoutOutput(farm).get_figure()
            plt.show()
            plt.close(ax.get_figure())

        algo = foxes.algorithms.Downwind(
            farm,
            states,
            rotor_model=args.rotor,
            wake_models=args.wakes,
            wake_frame="rotor_wd",
            partial_wakes=args.pwakes,
            mbook=mbook,
            verbosity=0,
        )

        outputs = [
            FV.D,
            FV.WD,
            FV.AMB_REWS,
            FV.REWS,
            FV.AMB_P,
            FV.P,
            FV.CT,
            FV.WEIGHT,
            FV.MAX_P,
        ]

        # run calculation with power mask:

        farm_results = algo.calc_farm(outputs=outputs)

        fr = farm_results.to_dataframe()
        print(fr[[FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P, FV.CT, FV.MAX_P]])

        o = foxes.output.FarmResultsEval(farm_results)
        P0 = o.calc_mean_farm_power(ambient=True)
        P = o.calc_mean_farm_power()
        print(f"\nFarm power: {P/1000:.1f} MW, Efficiency = {P/P0*100:.2f} %")

        o1 = foxes.output.StateTurbineMap(farm_results)

        # run calculation without power mask:

        mbook.finalize(algo)
        models.remove("set_Pmax")
        models.remove("PMask")

        farm_results = algo.calc_farm(outputs=outputs)

        fr = farm_results.to_dataframe()
        print(fr[[FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P, FV.CT, FV.MAX_P]])

        o = foxes.output.FarmResultsEval(farm_results)
        P0 = o.calc_mean_farm_power(ambient=True)
        P = o.calc_mean_farm_power()
        print(f"\nFarm power: {P/1000:.1f} MW, Efficiency = {P/P0*100:.2f} %")

        if not args.nofig:
            o0 = foxes.output.StateTurbineMap(farm_results)

            # show power:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            o0.plot_map(
                FV.P,
                ax=axs[0],
                edgecolor="white",
                title="Power, no power mask",
                cmap="YlOrRd",
                vmin=0,
                vmax=np.nanmax(Pmax_data),
            )
            o1.plot_map(
                FV.MAX_P,
                ax=axs[1],
                edgecolor="white",
                cmap="YlOrRd",
                title="Power mask",
                vmin=0,
                vmax=np.nanmax(Pmax_data),
            )
            o1.plot_map(
                FV.P,
                ax=axs[2],
                edgecolor="white",
                cmap="YlOrRd",
                title="Power, with power mask",
                vmin=0,
                vmax=np.nanmax(Pmax_data),
            )
            plt.show()
            plt.close(fig)

            # show ct:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            o0.plot_map(
                FV.CT,
                ax=axs[0],
                edgecolor="white",
                title="ct, no power mask",
                cmap="YlGn",
                vmin=0,
                vmax=1.0,
            )
            o1.plot_map(
                FV.MAX_P,
                ax=axs[1],
                edgecolor="white",
                cmap="YlOrRd",
                title="Power mask",
                vmin=0,
                vmax=np.nanmax(Pmax_data),
            )
            o1.plot_map(
                FV.CT,
                ax=axs[2],
                edgecolor="white",
                cmap="YlGn",
                title="ct, with power mask",
                vmin=0,
                vmax=1.0,
            )
            plt.show()
