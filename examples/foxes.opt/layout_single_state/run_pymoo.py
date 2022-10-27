import numpy as np
import argparse
import matplotlib.pyplot as plt
from iwopy.interfaces.pymoo import Optimizer_pymoo

import foxes
from foxes.opt.problems.layout import FarmLayoutOptProblem
from foxes.opt.constraints import FarmBoundaryConstraint, MinDistConstraint
from foxes.opt.objectives import MaxFarmPower

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nt", "--n_t", help="The number of turbines", type=int, default=10
    )
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
        help="Minimal turbine distance in unit D",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-A", "--opt_algo", help="The pymoo algorithm name", default="ga"
    )
    parser.add_argument(
        "-P", "--n_pop", help="The population size", type=int, default=50
    )
    parser.add_argument(
        "-G", "--n_gen", help="The nmber of generations", type=int, default=300
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

    boundary = foxes.utils.geom2d.Circle([0.0, 0.0], 1000.0)

    farm = foxes.WindFarm(boundary=boundary)
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=np.zeros(2),
        xy_step=np.array([50.0, 0.0]),
        n_turbines=args.n_t,
        turbine_models=["layout_opt", "kTI_02", ttype.name],
    )
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

        problem = FarmLayoutOptProblem("layout_opt", algo)
        problem.add_objective(MaxFarmPower(problem))
        problem.add_constraint(FarmBoundaryConstraint(problem))
        if args.min_dist is not None:
            problem.add_constraint(
                MinDistConstraint(problem, min_dist=args.min_dist, min_dist_unit="D")
            )
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
            term_pars=dict(
                type="default",
                n_max_gen=args.n_gen,
                ftol=1e-6,
                xtol=1e-6,
            ),
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

        fig, axs = plt.subplots(1, 2, figsize=(12, 8))

        foxes.output.FarmLayoutOutput(farm).get_figure(fig=fig, ax=axs[0])

        o = foxes.output.FlowPlots2D(algo, results.problem_results)
        p_min = np.array([-1100.0, -1100.0])
        p_max = np.array([1100.0, 1100.0])
        fig = o.get_mean_fig_horizontal(
            "WS",
            resolution=20,
            fig=fig,
            ax=axs[1],
            xmin=p_min[0],
            xmax=p_max[0],
            ymin=p_min[1],
            ymax=p_max[1],
        )
        dpars = dict(alpha=0.6, zorder=10, p_min=p_min, p_max=p_max)
        farm.boundary.add_to_figure(
            axs[1], fill_mode="outside_white", pars_distance=dpars
        )

        plt.show()
        plt.close(fig)
