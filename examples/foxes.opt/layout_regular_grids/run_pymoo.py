import numpy as np
import argparse
import matplotlib.pyplot as plt
from iwopy.interfaces.pymoo import Optimizer_pymoo

import foxes
from foxes.opt.problems.layout import RegGridsLayoutOptProblem
from foxes.opt.objectives import MaxNTurbines

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["Bastankhah_linear_k002"],
        nargs="+",
    )
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes model", default="auto"
    )
    parser.add_argument("--ws", help="The wind speed", type=float, default=9.0)
    parser.add_argument("--wd", help="The wind direction", type=float, default=270.0)
    parser.add_argument("--ti", help="The TI value", type=float, default=0.08)
    parser.add_argument("--rho", help="The air density", type=float, default=1.225)
    parser.add_argument(
        "-d",
        "--min_dist",
        help="Minimal turbine distance in m",
        type=float,
        default=500.0,
    )
    parser.add_argument("-m", "--n_maxr", help="Maximal turbines per row", type=int, default=None)
    parser.add_argument(
        "-A", "--opt_algo", help="The pymoo algorithm name", default="MixedVariableGA"
    )
    parser.add_argument(
        "-P", "--n_pop", help="The population size", type=int, default=50
    )
    parser.add_argument(
        "-G", "--n_gen", help="The number of generations", type=int, default=300
    )
    parser.add_argument(
        "-g", "--n_grids", help="The number of grids", type=int, default=2
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

    boundary = foxes.utils.geom2d.ClosedPolygon(np.array(
    [[0, 0], [0, 2100], [1500, 2300], [1400, -200]], dtype=np.float64)) \
        + foxes.utils.geom2d.Circle([2500.0, 0.0], 500.0) \
        + foxes.utils.geom2d.Circle([2500.0, 500.0], 600.0)

    farm = foxes.WindFarm(boundary=boundary)
    farm.add_turbine(foxes.Turbine(
        xy=np.array([0.0, 0.0]),
        turbine_models=["layout_opt", "kTI_02", ttype.name]
    ))

    states = foxes.input.states.SingleStateStates(
        ws=args.ws, wd=args.wd, ti=args.ti, rho=args.rho
    )

    algo = foxes.algorithms.Downwind(
        mbook,
        farm,
        states=states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame="rotor_wd",
        partial_wakes_model=args.pwakes,
        verbosity=0,
    )

    with foxes.utils.runners.DaskRunner(
        scheduler=args.scheduler,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        progress_bar=False,
        verbosity=1,
    ) as runner:

        problem = RegGridsLayoutOptProblem(
            "layout_opt", 
            algo, 
            min_spacing=args.min_dist,
            n_grids=args.n_grids,
            max_n_row=args.n_maxr,
            runner=runner,
            calc_farm_args={"ambient": True}
        )
        problem.add_objective(MaxNTurbines(problem))
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

        fig, ax = plt.subplots(figsize=(12, 8))
        foxes.output.FarmLayoutOutput(farm).get_figure(fig=fig, ax=ax)

        plt.show()
        plt.close(fig)
