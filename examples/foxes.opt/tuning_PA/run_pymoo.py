import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from iwopy.interfaces.pymoo import Optimizer_pymoo

import foxes
from foxes.opt.problems import OptFarmVars
from foxes.opt.objectives import MaxFarmPower, MinimalMaxTI, MinimalAPref
import foxes.variables as FV

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nt", "--n_t", help="The number of turbines", type=int, default=2
    )
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument(
        "-dx", "--deltax", help="The turbine distance in x", type=float, default=750.0
    )    
    parser.add_argument(
        "-dy", "--deltay", help="Turbine layout y step", type=float, default=0.0
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["PorteAgel_linear", "CrespoHernandez_quadratic"],
        nargs="+",
    )
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=["kTI_02"], nargs="+"
    )
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes model", default="auto"
    )
    parser.add_argument("--ws", help="The wind speed", type=float, default=8.0)
    parser.add_argument("--wd", help="The wind direction", type=float, default=270.0)
    parser.add_argument("--ti", help="The TI value", type=float, default=0.08)
    parser.add_argument("--rho", help="The air density", type=float, default=1.225)
    parser.add_argument(
        "-d",
        "--min_dist",
        help="Minimal turbine distance in unit D",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-A", "--opt_algo", help="The pymoo algorithm name", default="GA"
    )
    parser.add_argument(
        "-P", "--n_pop", help="The population size", type=int, default=80
    )
    parser.add_argument(
        "-G", "--n_gen", help="The nmber of generations", type=int, default=100
    )
    parser.add_argument(
        "-nop", "--no_pop", help="Switch off vectorization", action="store_true"
    )
    parser.add_argument("-sc", "--scheduler", help="The scheduler choice", default=None)
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
    args = parser.parse_args()

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=np.array([0.0, 0.0]),
        xy_step=np.array([args.deltax, args.deltay]),
        n_turbines=2,
        # turbine_models=args.tmodels + ["set_yawm", "yawm2yaw", ttype.name],
        turbine_models=args.tmodels + ["opt_yawm", "yawm2yaw", ttype.name],
    )

    # states = foxes.input.states.SingleStateStates(
    #     ws=args.ws, wd=args.wd, ti=args.ti, rho=args.rho
    # )

    path_local = '/home/sengers/data/foxes-cases/LES_ref/'
    df_in = pd.read_csv(path_local+'LES_in.csv')
    df_in.index = range(len(df_in))
    df_in['TI'] = df_in['TI']/100
    df_in = df_in.iloc[0,:].to_frame().T
    # yawm = [[k] for k in df_in['yawm']]

    states = foxes.input.states.StatesTable(
        data_source=df_in, ## df
        output_vars=[FV.WS, FV.WD, FV.SHEAR, FV.RHO, FV.TI],
        var2col={FV.WS: "WS", FV.SHEAR: "shear", FV.TI: "TI"},
        fixed_vars={FV.RHO: 1.225, FV.H: 90.0, FV.WD: 270.0},
        profiles={FV.WS: "ShearedProfile"},
    )
    df_out = pd.read_csv(path_local+'LES_out.csv')
    ref_vals = np.asarray([df_in['WS'][0],df_out['REWS'][0]]) ## Reference wind 
    print('ref_vals',ref_vals)

    algo = foxes.algorithms.Downwind(
        mbook,
        farm,
        states=states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame="yawed",
        partial_wakes_model='rotor_points',
        verbosity=0,
    )

    with foxes.utils.runners.DaskRunner(
        scheduler=args.scheduler,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        progress_bar=False,
        verbosity=1,
    ) as runner:

        problem = OptFarmVars("opt_yawm", algo, runner=runner)
        problem.add_var(FV.YAWM, float, 0., -40., 40., level="turbine")
        # problem.add_objective(MaxFarmPower(problem))
        # problem.add_objective(MinimalMaxTI(problem))
        problem.add_objective(MinimalAPref(problem,ref_vals=ref_vals))
        problem.initialize()

        solver = Optimizer_pymoo(
            problem,
            problem_pars=dict(
                vectorize=not args.no_pop,
            ),
            algo_pars=dict(
                type=args.opt_algo,
                pop_size=args.n_pop,
                seed=None,
            ),
            setup_pars=dict(),
            term_pars=('n_gen', args.n_gen),
        )
        solver.initialize()
        solver.print_info()

        ax = foxes.output.FarmLayoutOutput(farm).get_figure()
        plt.show()
        plt.close(ax.get_figure())

        results = solver.solve()
        solver.finalize(results)

        print()
        print(results)

        fr = results.problem_results.to_dataframe()
        print(fr[[FV.X, FV.Y, FV.AMB_WD, FV.REWS, FV.TI, FV.P, FV.YAWM]])

        o = foxes.output.FlowPlots2D(algo, results.problem_results)
        fig = o.get_mean_fig_xy("WS", resolution=10)
        plt.show()