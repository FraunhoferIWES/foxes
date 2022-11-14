import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV
from foxes.utils.runners import DaskRunner


def run_foxes(args):

    cks = None if args.nodask else {FV.STATE: args.chunksize}

    print("\nReading file", args.pmax_file)
    cols = [f"Pmax_{i}" for i in range(args.n_t)]
    Pmax_data = pd.read_csv(args.pmax_file, usecols=cols).to_numpy()

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype
    mbook.turbine_models["set_Pmax"] = foxes.models.turbine_models.SetFarmVars()
    mbook.turbine_models["set_Pmax"].add_var(FV.MAX_P, Pmax_data)
    models = args.tmodels + ["set_Pmax", ttype.name, "PMask"]

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

    if args.show_layout:
        ax = foxes.output.FarmLayoutOutput(farm).get_figure()
        plt.show()
        plt.close(ax.get_figure())

    algo = foxes.algorithms.Downwind(
        mbook,
        farm,
        states=states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame="rotor_wd",
        partial_wakes_model=args.pwakes,
        chunks=cks,
        verbosity=0,
    )

    # run calculation with power mask:

    farm_results = algo.calc_farm(vars_to_amb=[FV.REWS, FV.P])

    fr = farm_results.to_dataframe()
    print(fr[[FV.WD, FV.AMB_REWS, FV.REWS, FV.MAX_P, FV.AMB_P, FV.P]])

    o = foxes.output.FarmResultsEval(farm_results)
    P0 = o.calc_mean_farm_power(ambient=True)
    P = o.calc_mean_farm_power()
    print(f"\nFarm power: {P/1000:.1f} MW, Efficiency = {P/P0*100:.2f} %")

    o1 = foxes.output.StateTurbineMap(farm_results)

    # run calculation without power mask:

    algo.finalize(clear_mem=True)
    models.remove("set_Pmax")
    models.remove("PMask")

    farm_results = algo.calc_farm(vars_to_amb=[FV.REWS, FV.P])

    fr = farm_results.to_dataframe()
    print(fr[[FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]])

    o = foxes.output.FarmResultsEval(farm_results)
    P0 = o.calc_mean_farm_power(ambient=True)
    P = o.calc_mean_farm_power()
    print(f"\nFarm power: {P/1000:.1f} MW, Efficiency = {P/P0*100:.2f} %")

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
        "-p", "--pwakes", help="The partial wakes model", default="rotor_points"
    )
    parser.add_argument(
        "-f", "--pmax_file", help="The max_P csv file", default="power_mask.csv"
    )
    parser.add_argument(
        "-c", "--chunksize", help="The maximal chunk size", type=int, default=1000
    )
    parser.add_argument("-sc", "--scheduler", help="The scheduler choice", default=None)
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
        "-n",
        "--n_workers",
        help="The number of workers for distributed run",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-tw",
        "--threads_per_worker",
        help="The number of threads per worker for distributed run",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-sl",
        "--show_layout",
        help="Flag for showing layout figure",
        action="store_true",
    )
    parser.add_argument(
        "--nodask", help="Use numpy arrays instead of dask arrays", action="store_true"
    )
    args = parser.parse_args()

    with DaskRunner(
        scheduler=args.scheduler,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
    ) as runner:

        runner.run(run_foxes, args=(args,))
