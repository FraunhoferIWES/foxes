import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from iwopy.interfaces.pymoo import Optimizer_pymoo

import foxes
from foxes.utils import wd2wdvec, wd2uv
from foxes.opt.problems import OptFarmVars
import foxes.variables as FV
import foxes.constants as FC
from iwopy import Objective

def mask_turb(y,z,rcy=0,hh=0,rr=63):
    y = np.array(y)
    z = np.array(z)
    ## Make mask of rotor area virtual downstream turbine
    mask = np.zeros((len(z),len(y)))
    for z_idx in range(len(mask)): ## loop over vertical
        z_tmp = z[z_idx] 
        y_left = rcy - np.sqrt(rr**2-(z_tmp-hh)**2) ## use equation of circle
        y_right = rcy + np.sqrt(rr**2-(z_tmp-hh)**2) ## to determine left,right edge 
        for y_idx in range(len(mask[z_idx])): ## loop over horizontal
            y_tmp = y[y_idx]
            if y_left < y_tmp < y_right: ## if between left and right edge
                mask[z_idx,y_idx] = 1   
            else:
                mask[z_idx,y_idx] = np.nan
    return mask

class MinimalAPref(Objective):
    """
    Minimize the AP diff to reference

    Parameters
    ----------
    problem : foxes.opt.FarmOptProblem
        The underlying optimization problem
    name : str
        The name of the objective function
    kwargs : dict, optional
        Additional parameters for `FarmVarObjective`

    """

    def __init__(self, problem, ref_vals, g_pts, name="minimize_APref", **kwargs):
        self.ref_vals = ref_vals ## DEV
        self.g_pts = g_pts ## DEV

        super().__init__(problem, name, vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float())

    def n_components(self):
        return 1
    
    def maximize(self):
        return [False]


    def calc_individual(self, vars_int, vars_float, problem_results, components=None):
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)
        problem_results : Any
            The results of the variable application
            to the problem
        components : list of int, optional
            The selected components or None for all

        Returns
        -------
        values : np.array
            The component values, shape: (n_sel_components,)

        """
        point_results = problem.algo.calc_points(problem_results, points=self.g_pts)
        data = point_results[FV.WS].to_numpy()
        data = np.nanmean(data,axis=1)
        
        del point_results
        data = np.sum(np.abs(data - self.ref_vals[None,:]))
        return np.array([data], dtype=np.float64)

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
        "-G", "--n_gen", help="The number of generations", type=int, default=100
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
    farm.add_turbine(
        foxes.Turbine(
            xy=np.array([0.0, 0.0]),
            D = 126.,
            H = 90.,
            turbine_models=args.tmodels + ["opt_yawm", "yawm2yaw", ttype.name],
        )
    )

    path_local = '/home/sengers/data/foxes-cases/LES_ref/'
    df_in = pd.read_csv(path_local+'LES_in.csv')
    df_in.index = range(len(df_in))
    df_in['TI'] = df_in['TI']/100
    df_in = df_in.iloc[0,:].to_frame().T ## First case only
    # yawm = [[k] for k in df_in['yawm']]

    states = foxes.input.states.StatesTable(
        data_source=df_in, ## df
        output_vars=[FV.WS, FV.WD, FV.SHEAR, FV.RHO, FV.TI],
        var2col={FV.WS: "WS", FV.SHEAR: "shear", FV.TI: "TI"},
        fixed_vars={FV.RHO: 1.225, FV.H: 90.0, FV.WD: 270.0},
        profiles={FV.WS: "ShearedProfile"},
    )

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

    ## Define other variables needed for optimization
    df_out = pd.read_csv(path_local+'LES_out.csv')
    ref_vals = np.asarray([df_out['REWS'][0]]) ## Reference wind; first case only

    ## Define points
    x_pos = [750]
    y_pos = np.arange(farm.turbines[0].D/2*-1,farm.turbines[0].D/2,5)
    z_pos = np.arange(farm.turbines[0].H-farm.turbines[0].D/2,farm.turbines[0].H+farm.turbines[0].D/2,5)

    print(farm.turbines[0].xy)

    try:
        n_x = np.append(wd2uv(states.fixed_vars['WD']), [0.0], axis=0) 
        n_z = np.array([0.0, 0.0, 1.0])
        n_y = np.cross(n_z, n_x)    
    except KeyError:
        raise KeyError('WD not found in fixed_vars')

    n_states = len(df_in)
    N_x, N_y, N_z = len(x_pos), len(y_pos), len(z_pos)
    n_pts = N_x*N_y*N_z
    g_pts = np.zeros((n_states,N_x, N_y, N_z, 3), dtype=FC.DTYPE)
    for ix in range(N_x):
        x_pos_tmp = x_pos[ix]
        g_pts[:] += x_pos_tmp * n_x[None, None, None, None, :]
        g_pts[:] += y_pos[None, None, :, None, None] * n_y[None, None, None, None, :]
        g_pts[:] += z_pos[None, None, None, :, None] * n_z[None, None, None, None, :]
    mask = mask_turb(y_pos,z_pos,farm.turbines[0].xy[1],farm.turbines[0].H,farm.turbines[0].D/2) ## circular rotor area
    mask = np.swapaxes(mask,0,1) ## correct order of dimensions
    g_pts = g_pts*mask[None,None,:,:,None] ## mask out
    g_pts = g_pts.reshape(n_states, n_pts, 3)

    with foxes.utils.runners.DaskRunner(
        scheduler=args.scheduler,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        progress_bar=False,
        verbosity=1,
    ) as runner:

        problem = OptFarmVars("opt_yawm", algo, runner=runner)
        problem.add_var(FV.YAWM, float, 0., -40., 40., level="turbine")
        problem.add_objective(MinimalAPref(problem,ref_vals=ref_vals,g_pts=g_pts))
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