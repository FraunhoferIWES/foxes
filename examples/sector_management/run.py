import time
import argparse
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV

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
        help="The states input file (path or static)",
        default="wind_rose_bremen.csv",
    )
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument(
        "-sm",
        "--sector_file",
        help="The sector management file",
        default="sector_rules.csv",
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes models", default=None, nargs="+"
    )
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["Bastankhah2014_linear_k002"],
        nargs="+",
    )
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=[], nargs="+"
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

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    mbook.turbine_models["sector_rules"] = foxes.models.turbine_models.SectorManagement(
        data_source=args.sector_file,
        col_tnames="tname",
        range_vars=[FV.WD, FV.REWS],
        target_vars=[FV.MAX_P],
        colmap={"REWS_min": "WS_min", "REWS_max": "WS_max"},
    )

    states = foxes.input.states.StatesTable(
        data_source=args.states,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.WEIGHT: "weight"},
        fixed_vars={FV.RHO: 1.225, FV.TI: 0.05},
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=[0.0, 0.0],
        xy_step=[600.0, 0.0],
        n_turbines=2,
        turbine_models=args.tmodels + [ttype.name, "sector_rules", "PMask"],
    )

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame="rotor_wd",
        partial_wakes=args.pwakes,
        mbook=mbook,
        engine=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
    )

    outputs = [
        FV.D,
        FV.WD,
        FV.AMB_WD,
        FV.H,
        FV.AMB_REWS,
        FV.REWS,
        FV.AMB_P,
        FV.MAX_P,
        FV.P,
        FV.CT,
        FV.WEIGHT,
    ]

    time0 = time.time()
    farm_results = algo.calc_farm(outputs=outputs)
    time1 = time.time()

    print("\nCalc time =", time1 - time0, "\n")

    print(farm_results)

    fr = farm_results.to_dataframe()
    sel = fr[FV.MAX_P].dropna().index
    fr = fr.loc[sel]
    print(fr)

    if not args.nofig:
        o = foxes.output.RosePlotOutput(farm_results)
        fig = o.get_figure(
            16,
            FV.P,
            [100, 1000, 2000, 4000, 5001, 7000],
            turbine=0,
            title="Power turbine 0",
            figsize=(12, 6),
            rect=[0.05, 0.1, 0.4, 0.8],
        )

        o = foxes.output.RosePlotOutput(farm_results)
        fig = o.get_figure(
            16,
            FV.P,
            [100, 1000, 2000, 4000, 5001, 7000],
            turbine=1,
            title="Power turbine 1",
            fig=fig,
            rect=[0.35, 0.1, 0.8, 0.8],
        )
        plt.show()
