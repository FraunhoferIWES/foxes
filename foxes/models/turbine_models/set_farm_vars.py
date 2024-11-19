import numpy as np

from foxes.core import TurbineModel
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC


class SetFarmVars(TurbineModel):
    """
    Set farm data variables to given data.

    Attributes
    ----------
    vars: list of str
        The variables to be set
    once: bool
        Flag for running only once

    :group: models.turbine_models

    """

    def __init__(self, pre_rotor=False, once=False):
        """
        Constructor.

        Parameters
        ----------
        pre_rotor: bool
            Flag for running this model before
            running the rotor model.
        once: bool
            Flag for running only once

        """
        super().__init__(pre_rotor=pre_rotor)
        self.once = once
        self.reset()

    def add_var(self, var, data):
        """
        Add data for a variable.

        Parameters
        ----------
        var: str
            The variable name
        data: numpy.ndarray
            The data, shape: (n_states, n_turbines)

        """
        if self.initialized:
            raise ValueError(
                f"Model '{self.name}': Cannot add_var after initialization"
            )
        if self.running:
            raise ValueError(f"Model '{self.name}': Cannot add_var while running")
        self.vars.append(var)
        self.__vdata.append(np.asarray(data, dtype=config.dtype_double))

    def reset(self):
        """
        Remove all variables.
        """
        if self.running:
            raise ValueError(f"Model '{self.name}': Cannot reset while running")
        self.vars = []
        self.__vdata = []

    def initialize(self, algo, verbosity=0, force=False):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent
        force: bool
            Overwrite existing data

        """
        super().initialize(algo, verbosity, force)
        self.__once_done = set()

    def output_farm_vars(self, algo):
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        output_vars: list of str
            The output variable names

        """
        return self.vars

    def load_data(self, algo, verbosity=0):
        """
        Load and/or create all model data that is subject to chunking.

        Such data should not be stored under self, for memory reasons. The
        data returned here will automatically be chunked and then provided
        as part of the mdata object during calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        idata: dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """
        idata = super().load_data(algo, verbosity)

        for i, v in enumerate(self.vars):
            data = np.full(
                (algo.n_states, algo.n_turbines), np.nan, dtype=config.dtype_double
            )
            vdata = self.__vdata[i]

            # handle special case of call during vectorized optimization:
            if (
                np.ndim(vdata)
                and vdata.shape[0] != algo.n_states
                and hasattr(algo.states, "n_pop")
            ):
                n_pop = algo.states.n_pop
                n_ost = algo.states.states.size()
                n_trb = algo.n_turbines
                vdata = np.zeros((n_pop, n_ost, n_trb), dtype=config.dtype_double)
                vdata[:] = self.__vdata[i][None, :]
                vdata = vdata.reshape(n_pop * n_ost, n_trb)

            data[:] = vdata
            idata["data_vars"][self.var(v)] = ((FC.STATE, FC.TURBINE), data)

            # special case of turbine positions:
            if v in [FV.X, FV.Y]:
                i = [FV.X, FV.Y].index(v)
                for ti in range(algo.n_turbines):
                    t = algo.farm.turbines[ti]
                    if len(t.xy.shape) == 1:
                        xy = np.zeros((algo.n_states, 2), dtype=config.dtype_double)
                        xy[:] = t.xy[None, :]
                        t.xy = xy
                    t.xy[:, i] = np.where(
                        np.isnan(data[:, ti]), t.xy[:, i], data[:, ti]
                    )

            # special case of rotor diameter and hub height:
            if v in [FV.D, FV.H]:
                for ti in range(algo.n_turbines):
                    t = algo.farm.turbines[ti]
                    x = np.zeros(algo.n_states, dtype=config.dtype_double)
                    if v == FV.D:
                        x[:] = t.D
                        t.D = x
                    else:
                        x[:] = t.H
                        t.H = x
                    x[:] = np.where(np.isnan(data[:, ti]), x, data[:, ti])

        return idata

    def set_running(
        self,
        algo,
        data_stash,
        sel=None,
        isel=None,
        verbosity=0,
    ):
        """
        Sets this model status to running, and moves
        all large data to stash.

        The stashed data will be returned by the
        unset_running() function after running calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data_stash: dict
            Large data stash, this function adds data here.
            Key: model name. Value: dict, large model data
        sel: dict, optional
            The subset selection dictionary
        isel: dict, optional
            The index subset selection dictionary
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().set_running(algo, data_stash, sel, isel, verbosity)

        data_stash[self.name]["vdata"] = self.__vdata
        del self.__vdata

    def unset_running(
        self,
        algo,
        data_stash,
        sel=None,
        isel=None,
        verbosity=0,
    ):
        """
        Sets this model status to not running, recovering large data
        from stash

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data_stash: dict
            Large data stash, this function adds data here.
            Key: model name. Value: dict, large model data
        sel: dict, optional
            The subset selection dictionary
        isel: dict, optional
            The index subset selection dictionary
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().unset_running(algo, data_stash, sel, isel, verbosity)
        self.__vdata = data_stash[self.name].pop("vdata")

    def calculate(self, algo, mdata, fdata, st_sel):
        """
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        st_sel: slice or numpy.ndarray of bool
            The state-turbine selection,
            for shape: (n_states, n_turbines)

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        i0 = mdata.states_i0(counter=True)
        if not self.once or i0 not in self.__once_done:

            if self.pre_rotor:
                order = np.s_[:]
                ssel = np.s_[:]
            else:
                order = fdata[FV.ORDER]
                ssel = fdata[FV.ORDER_SSEL]

            bsel = np.zeros((fdata.n_states, fdata.n_turbines), dtype=bool)
            bsel[st_sel] = True

            for v in self.vars:
                data = mdata[self.var(v)][ssel, order]
                hsel = ~np.isnan(data)
                tsel = bsel & hsel

                fdata[v][tsel] = data[tsel]

            self.__once_done.add(i0)

        return {v: fdata[v] for v in self.vars}
