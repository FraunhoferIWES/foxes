import numpy as np
import argparse
import matplotlib.pyplot as plt
from iwopy import LocalFD
from iwopy.interfaces.pygmo import Optimizer_pygmo

import foxes
import foxes.variables as FV
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
    parser.add_argument(
        "-s",
        "--states",
        help="The states input file (path or static)",
        default="wind_rose_bremen.csv",
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
    states = foxes.input.states.StatesTable(
        data_source=args.states,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.WEIGHT: "weight"},
        fixed_vars={FV.RHO: args.rho, FV.TI: args.ti},
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
    problem.add_constraint(FarmBoundaryConstraint(problem, tol=1.))
    if args.min_dist is not None:
        problem.add_constraint(MinDistConstraint(problem, min_dist=args.min_dist, min_dist_unit="D"))
    gproblem = LocalFD(problem, deltas=1., fd_order=args.fd_order)
    gproblem.initialize()
    
    solver = Optimizer_pygmo(
        gproblem,
        problem_pars=dict(pop=not args.no_pop),
        algo_pars=dict(type="ipopt", tol=1e-3),
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

    ax = foxes.output.FarmLayoutOutput(farm).get_figure()
    plt.show()
    plt.close(ax.get_figure())

    o = foxes.output.FlowPlots2D(algo, results.problem_results)
    fig = o.get_mean_fig_horizontal("WS", resolution=20, xmin=-1100, xmax=1100, ymin=-1100, ymax=1100)
    farm.boundary.add_to_figure(fig.axes[0])
    plt.show()
    plt.close(fig)
