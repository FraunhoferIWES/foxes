import time
import argparse
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV
from foxes.utils.runners import DaskRunner


def run_foxes(args):

    cks = None if args.nodask else {FV.STATE: args.chunksize}

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    states = foxes.input.states.Timeseries(
        data_source=args.states,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
        fixed_vars={FV.RHO: 1.225},
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_from_file(
        farm, args.layout, turbine_models=args.tmodels + [ttype.name]
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
    )

    time0 = time.time()

    farm_results = algo.calc_farm(vars_to_amb=[FV.REWS, FV.P])

    time1 = time.time()
    print("\nCalc time =", time1 - time0, "\n")

    print(farm_results, "\n")

    fr = farm_results.to_dataframe()
    print(fr[[FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]])

    o = foxes.output.FarmResultsEval(farm_results)
    P0 = o.calc_mean_farm_power(ambient=True)
    P = o.calc_mean_farm_power()

    print(f"\nFarm power: {P/1000:.1f} MW, Efficiency = {P/P0*100:.2f} %")
    
    # get timestep and duration of timeseries for yield and capacity calculations
    ts= o.calc_times(states) # in hours

    # add yield results by state and turbine
    o.calc_yield(timestep=ts)
    o.calc_yield(timestep=ts, ambient=True)

    turbine_yield = o.calc_turbine_yield()
    print("Yield values by turbine:")
    print(turbine_yield) # in GWh
    print()

    turbine_yield_ann = o.calc_turbine_yield(annual=True)
    print("\nAnnual yield values by turbine [GWh]:")
    print(turbine_yield_ann * 1e-6) # in GWh
    print()

    # add capacity to farm results
    o.calc_capacity(P_nom=ttype.P_nominal)
    o.calc_capacity(P_nom=ttype.P_nominal, ambient=True)
    
    # calculate farm yield, P75 and P90
    farm_yield, P75, P90 = o.calc_farm_yield()
    print(f"\nFarm yield is {farm_yield:.2f} GWh")
    print(f"Farm P75 is {P75:.2f} GWh")
    print(f"Farm P90 is {P90:.2f} GWh")

    # add efficiency to farm results
    o.calc_efficiency()

    # efficiency by turbine
    turbine_eff = o.calc_states_mean([FV.EFF]) ## all results are NaN due some zero AMB_P values
    print("Efficiency by turbine:")
    print(turbine_eff) 

    farm_df = farm_results.to_dataframe()
    print("\nFarm results data:")
    print(farm_df[[FV.P, FV.AMB_P, FV.EFF, FV.CAP, FV.AMB_CAP, FV.YLD, FV.AMB_YLD]])
    print()


if __name__ == "__main__":

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
        default="timeseries.csv.gz",
    )
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes model", default="rotor_points"
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
    parser.add_argument(
        "-ts",
        "--timestep",
        help="The timestep of the input timeseries or data in minutes",
        default = 60*24*365 # default is one year
    )
    args = parser.parse_args()

    # set timestep for debugging
    args.timestep = 30

    with DaskRunner(
        scheduler=args.scheduler,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
    ) as runner:

        runner.run(run_foxes, args=(args,))
