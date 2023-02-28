import numpy as np
import argparse
import matplotlib.pyplot as plt
from iwopy.interfaces.pymoo import Optimizer_pymoo

import foxes
import foxes.opt.problems.layout.geom_layouts as grg

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--min_dist",
        help="Minimal turbine distance in m",
        type=float,
        default=400.0,
    )
    parser.add_argument("-m", "--n_maxr", help="Maximal turbines per row", type=int, default=None)
    parser.add_argument(
        "-g", "--n_grids", help="The number of grids", type=int, default=2
    )
    parser.add_argument(
        "-nop", "--no_pop", help="Switch off vectorization", action="store_true"
    )
    parser.add_argument(
        "-A", "--opt_algo", help="The pymoo algorithm name", default="MixedVariableGA"
    )
    parser.add_argument(
        "-P", "--n_pop", help="The population size", type=int, default=100
    )
    parser.add_argument(
        "-G", "--n_gen", help="The number of generations", type=int, default=150
    )
    args = parser.parse_args()

    boundary = foxes.utils.geom2d.ClosedPolygon(np.array(
    [[0, 1000], [400, 1600], [2500, 1100], [2200, 300]], dtype=np.float64)) \
        + foxes.utils.geom2d.Circle([2500.0, 0.0], 500.0) \
        + foxes.utils.geom2d.ClosedPolygon(np.array(
    [[2200, 0], [2200, 2600], [2800, 2600], [2800, 0]], dtype=np.float64))

    problem = grg.GeomRegGrids(boundary, args.min_dist, args.n_grids, args.n_maxr)
    problem.add_objective(grg.OMaxN(problem))
    problem.initialize()

    fig = problem.get_fig().get_figure()
    plt.show()
    plt.close(fig)

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

    results = solver.solve()
    solver.finalize(results)

    print()
    print(results)

    xy, valid = results.problem_results
    fig = problem.get_fig(xy, valid).get_figure()
    plt.show()
    plt.close(fig)