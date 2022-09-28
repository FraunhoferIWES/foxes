import numpy as np
import argparse
import matplotlib.pyplot as plt
from iwopy import DiscretizeRegGrid
from iwopy.interfaces.pygmo import Optimizer_pygmo

import foxes
from foxes.opt.problems.layout import FarmLayoutOptProblem
from foxes.opt.constraints import FarmBoundaryConstraint, MinDistConstraint
from foxes.opt.objectives import MaxFarmPower

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nt", "--n_t", help="The number of turbines", type=int, default=5
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
        default=["CrespoHernandez_quadratic", "Bastankhah_linear"],
        nargs="+",
    )
    parser.add_argument(
        "-p", "--pwakes", help="The partial wakes model", default="auto"
    )
    parser.add_argument("--ws", help="The wind speed", type=float, default=9.0)
    parser.add_argument("--wd", help="The wind direction", type=float, default=270.0)
    parser.add_argument("--ti", help="The TI value", type=float, default=0.08)
    parser.add_argument("--rho", help="The air density", type=float, default=1.225)
    parser.add_argument("-d", "--min_dist", help="Minimal turbine distance in unit D", type=float, default=None)
    parser.add_argument("-nop", "--no_pop", help="Switch off vectorization", action="store_true")
    parser.add_argument("-O", "--fd_order", help="Finite difference derivative order", type=int, default=1)
    args = parser.parse_args()

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    boundary = foxes.utils.geom2d.Circle([0., 0.], 1000.)

    farm = foxes.WindFarm(boundary=boundary)
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=np.zeros(2),
        xy_step=np.array([50.0, 5.0]),
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

    problem = FarmLayoutOptProblem("layout_opt", algo)
    problem.add_objective(MaxFarmPower(problem))
    problem.add_constraint(FarmBoundaryConstraint(problem, tol=0.5))
    if args.min_dist is not None:
        problem.add_constraint(MinDistConstraint(problem, min_dist=args.min_dist, min_dist_unit="D"))
    gproblem = DiscretizeRegGrid(
        problem, deltas=0.1, fd_order=args.fd_order, fd_bounds_order=1, tol=1e-6
    )
    gproblem.initialize()
    
    solver = Optimizer_pygmo(
        gproblem,
        problem_pars=dict(pop=not args.no_pop),
        algo_pars=dict(type="ipopt"),
    )
    """
    solver = Optimizer_pygmo(
        gproblem,
        problem_pars=dict(pop=True),
        algo_pars=dict(type="nlopt", optimizer="ccsaq", ftol_rel=0, ftol_abs=1e-4, xtol_rel=0,xtol_abs=0),
    )
    """
    solver.initialize()
    solver.print_info()

    ax = foxes.output.FarmLayoutOutput(farm).get_figure()
    plt.show()
    plt.close(ax.get_figure())

    results = solver.solve()
    solver.finalize(results)

    print()
    print(results)

    ax = foxes.output.FarmLayoutOutput(farm).get_figure()
    plt.show()
    plt.close(ax.get_figure())

    o = foxes.output.FlowPlots2D(algo, results.problem_results)
    g = o.gen_states_fig_horizontal("WS", resolution=10, xmin=-1100, xmax=1100, ymin=-1100, ymax=1100)
    fig = next(g)
    farm.boundary.add_to_figure(fig.axes[0])
    plt.show()
    plt.close(fig)
