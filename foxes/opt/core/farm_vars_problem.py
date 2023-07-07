import numpy as np
from abc import abstractmethod

from .farm_opt_problem import FarmOptProblem
from foxes.models.turbine_models import SetFarmVars
import foxes.constants as FC


class FarmVarsProblem(FarmOptProblem):
    """
    Abstract base class for models that optimize
    farm variables.

    :group: opt.core

    """

    def initialize(self, pre_rotor_vars, post_rotor_vars, verbosity=1, **kwargs):
        """
        Initialize the object.

        Parameters
        ----------
        pre_rotor_vars: list of str or dict
            The pre_rotor farm variables. If dict, then
            key: sub-model str, value: var names as list of str
        post_rotor_vars: list of str or dict
            The post_rotor farm variables. If dict, then
            key: sub-model str, value: var names as list of str
        verbosity: int
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Additional parameters for super class init

        """
        self._vars_pre = {}
        self._vars_post = {}
        if isinstance(pre_rotor_vars, dict):
            self._vars_pre = {m: v for m, v in pre_rotor_vars.items() if len(v)}
        elif len(pre_rotor_vars):
            self._vars_pre = {self.name: pre_rotor_vars}
        if isinstance(post_rotor_vars, dict):
            self._vars_post = {m: v for m, v in post_rotor_vars.items() if len(v)}
        elif len(post_rotor_vars):
            self._vars_post = {self.name: post_rotor_vars}

        cnt = 0
        for src, pre in zip((self._vars_pre, self._vars_post), (True, False)):
            for mname, vrs in src.items():
                if mname in self.algo.mbook.turbine_models:
                    m = self.algo.mbook.turbine_models[mname]
                    if not isinstance(m, SetFarmVars):
                        raise KeyError(
                            f"FarmOptProblem '{self.name}': Turbine model entry '{mname}' already exists in model book, and is not of type SetFarmVars"
                        )
                    elif m.pre_rotor != pre:
                        raise ValueError(
                            f"FarmOptProblem '{self.name}': Turbine model entry '{mname}' exists in model book, and disagrees on pre_rotor = {pre}"
                        )
                else:
                    self.algo.mbook.turbine_models[mname] = SetFarmVars(pre_rotor=pre)

                found = False
                for t in self.algo.farm.turbines:
                    if mname in t.models:
                        found = True
                        break
                if not found:
                    raise ValueError(
                        f"FarmOptProblem '{self.name}': Missing entry '{mname}' among any of the turbine models"
                    )
                cnt += len(vrs)
        if not cnt:
            raise ValueError(
                f"Problem '{self.name}': Neither pre_rotor_vars not post_rotor_vars containing variables"
            )

        super().initialize(verbosity=verbosity, **kwargs)

    @abstractmethod
    def opt2farm_vars_individual(self, vars_int, vars_float):
        """
        Translates optimization variables to farm variables

        Parameters
        ----------
        vars_int: numpy.ndarray
            The integer optimization variable values,
            shape: (n_vars_int,)
        vars_float: numpy.ndarray
            The float optimization variable values,
            shape: (n_vars_float,)

        Returns
        -------
        farm_vars: dict
            The foxes farm variables. Key: var name,
            value: numpy.ndarray with values, shape:
            (n_states, n_sel_turbines)

        """
        pass

    @abstractmethod
    def opt2farm_vars_population(self, vars_int, vars_float, n_states):
        """
        Translates optimization variables to farm variables

        Parameters
        ----------
        vars_int: numpy.ndarray
            The integer optimization variable values,
            shape: (n_pop, n_vars_int)
        vars_float: numpy.ndarray
            The float optimization variable values,
            shape: (n_pop, n_vars_float)
        n_states: int
            The number of original (non-pop) states

        Returns
        -------
        farm_vars: dict
            The foxes farm variables. Key: var name,
            value: numpy.ndarray with values, shape:
            (n_pop, n_states, n_sel_turbines)

        """
        pass

    def update_problem_individual(self, vars_int, vars_float):
        """
        Update the algo and other data using
        the latest optimization variables.

        This function is called before running the farm
        calculation.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float: np.array
            The float variable values, shape: (n_vars_float,)

        """
        super().update_problem_individual(vars_int, vars_float)

        # prepare:
        n_states = self._org_n_states
        fvars = self.opt2farm_vars_individual(vars_int, vars_float)

        # update turbine model that sets vars to opt values:
        for src in (self._vars_pre, self._vars_post):
            for mname, vrs in src.items():
                model = self.algo.mbook.turbine_models[mname]
                model.reset()
                for v in vrs:
                    vals = fvars.pop(v)
                    if self.all_turbines:
                        model.add_var(v, vals)
                    else:
                        data = np.zeros(
                            (n_states, self.algo.n_turbines), dtype=FC.DTYPE
                        )
                        data[:, self.sel_turbines] = vals
                        model.add_var(v, data)

        if len(fvars):
            raise KeyError(
                f"Problem '{self.name}': Too many farm vars from opt2farm_vars_individual: {list(fvars.keys())}"
            )

    def update_problem_population(self, vars_int, vars_float):
        """
        Update the algo and other data using
        the latest optimization variables.

        This function is called before running the farm
        calculation.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_pop, n_vars_int,)
        vars_float: np.array
            The float variable values, shape: (n_pop, n_vars_float,)

        """
        super().update_problem_population(vars_int, vars_float)

        # prepare:
        n_pop = len(vars_float)
        n_states = self._org_n_states
        n_pstates = n_states * n_pop
        fvars = self.opt2farm_vars_population(vars_int, vars_float, n_states)

        # update turbine model that sets vars to opt values:
        for src in (self._vars_pre, self._vars_post):
            for mname, vrs in src.items():
                model = self.algo.mbook.turbine_models[mname]
                model.reset()
                for v in vrs:
                    vals = fvars.pop(v)
                    shp0 = list(vals.shape)
                    shp1 = [n_pstates] + shp0[2:]
                    if self.all_turbines:
                        model.add_var(v, vals.reshape(shp1))
                    else:
                        data = np.zeros(
                            (n_pstates, self.algo.n_turbines), dtype=FC.DTYPE
                        )
                        data[:, self.sel_turbines] = vals.reshape(shp1)
                        model.add_var(v, data)
                        del data

        if len(fvars):
            raise KeyError(
                f"Problem '{self.name}': Too many farm vars from opt2farm_vars_population: {list(fvars.keys())}"
            )
