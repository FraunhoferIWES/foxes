import numpy as np

from foxes.opt.core import FarmOptProblem
import foxes.variables as FV
import foxes.constants as FC
from foxes.models.wake_frames import YawedWakes

class TuningProblem(FarmOptProblem):
    """
    Tuning the parameters in Porte-Agel

    Parameters
    ----------
    name : str
        The problem's name
    algo : foxes.core.Algorithm
        The algorithm
    runner : foxes.core.Runner, optional
        The runner for running the algorithm
    sel_turbines : list of int, optional
        The turbines selected for optimization,
        or None for all
    calc_farm_args : dict
        Additional parameters for algo.calc_farm()
    kwargs : dict, optional
        Additional parameters for `FarmOptProblem`

    """

    def __init__(
        self,
        name,
        algo,
        runner=None,
        # sel_turbines=None,
        calc_farm_args={},
        **kwargs,
    ):
        super().__init__(
            name,
            algo,
            runner,
            pre_rotor=True,
            # sel_turbines=sel_turbines,
            calc_farm_args=calc_farm_args,
            **kwargs,
        )

    def var_names_float(self):
        """
        The names of float variables.

        Returns
        -------
        names : list of str
            The names of the float variables

        """
        return ['PA_ALPHA','PA_BETA'] ## hardcoded

    def output_farm_vars(self, algo):
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        output_vars : list of str
            The output variable names

        """
        return [FV.PA_ALPHA, FV.PA_BETA] ## hardcoded
    

    def opt2farm_vars_individual(self, vars_int, vars_float):
        """
        Translates optimization variables to farm variables

        Parameters
        ----------
        vars_int : numpy.ndarray
            The integer optimization variable values,
            shape: (n_vars_int,)
        vars_float : numpy.ndarray
            The float optimization variable values,
            shape: (n_vars_float,)

        Returns
        -------
        farm_vars : dict
            The foxes farm variables. Key: var name,
            value: numpy.ndarray with values, shape:
            (n_states, n_sel_turbines)

        """
        farm_vars = {
            FV.PA_ALPHA: np.zeros((self.algo.n_states), dtype=FC.DTYPE),
            FV.PA_ALPHA: np.zeros((self.algo.n_states), dtype=FC.DTYPE),
        }
        farm_vars[FV.PA_ALPHA][:] = vars_float[0]
        farm_vars[FV.PA_BETA][:] = vars_float[1]
        return farm_vars


    def opt2farm_vars_population(self, vars_int, vars_float, n_states):
        """
        Translates optimization variables to farm variables

        Parameters
        ----------
        vars_int : numpy.ndarray
            The integer optimization variable values,
            shape: (n_pop, n_vars_int)
        vars_float : numpy.ndarray
            The float optimization variable values,
            shape: (n_pop, n_vars_float)
        n_states : int
            The number of original (non-pop) states

        Returns
        -------
        farm_vars : dict
            The foxes farm variables. Key: var name,
            value: numpy.ndarray with values, shape:
            (n_pop, n_states, n_sel_turbines)

        """
        pass



    def finalize_individual(self, vars_int, vars_float, verbosity=1):
        """
        Finalization, given the champion data.

        Parameters
        ----------
        vars_int : np.array
            The optimal integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The optimal float variable values, shape: (n_vars_float,)
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        problem_results : Any
            The results of the variable application
            to the problem
        objs : np.array
            The objective function values, shape: (n_objectives,)
        cons : np.array
            The constraints values, shape: (n_constraints,)

        """
        res, objs, cons = super().finalize_individual(vars_int, vars_float, verbosity)

        self.mbook.wake_frames["yawed_tuning"] = YawedWakes(
            alpha=vars_float[0],beta=vars_float[1]) ## add new wake_frame (deflection model) with some initial values

        return res, objs, cons
