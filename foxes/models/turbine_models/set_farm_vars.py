import numpy as np

from foxes.core import TurbineModel
import foxes.constants as FC
import foxes.variables as FV


class SetFarmVars(TurbineModel):
    """
    Set farm data variables to given data.

    Attributes
    ----------
    vars: list of str
        The variables to be set

    :group: models.turbine_models

    """

    def __init__(self, pre_rotor=False):
        """
        Constructor.

        Parameters
        ----------
        pre_rotor: bool
            Flag for running this model before
            running the rotor model.

        """
        super().__init__(pre_rotor=pre_rotor)
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
        self.vars.append(var)
        self._vdata.append(np.asarray(data, dtype=FC.DTYPE))

    def reset(self):
        """
        Remove all variables.
        """
        self.vars = []
        self._vdata = []

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
            data = np.full((algo.n_states, algo.n_turbines), np.nan, dtype=FC.DTYPE)
            vdata = self._vdata[i]

            # handle special case of call during vectorized optimization:
            if (
                np.ndim(vdata)
                and vdata.shape[0] != algo.n_states
                and hasattr(algo.states, "n_pop")
            ):
                n_pop = algo.states.n_pop
                n_ost = algo.states.states.size()
                n_trb = algo.n_turbines
                vdata = np.zeros((n_pop, n_ost, n_trb), dtype=FC.DTYPE)
                vdata[:] = self._vdata[i][None, :]
                vdata = vdata.reshape(n_pop * n_ost, n_trb)

            data[:] = vdata
            idata["data_vars"][self.var(v)] = ((FC.STATE, FC.TURBINE), data)

        return idata

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

            # special case of turbine positions:
            if v in [FV.X, FV.Y]:
                i = [FV.X, FV.Y].index(v)
                for ti in np.where(tsel)[1]:
                    t = algo.farm.turbines[ti]
                    if len(t.xy.shape) == 1:
                        xy = np.zeros((algo.n_states, 2), dtype=FC.DTYPE)
                        xy[:] = t.xy[None, :]
                        t.xy = xy
                    i0 = fdata.states_i0()
                    hsel = tsel[:, ti]
                    ssel = i0 + np.where(hsel)[0]
                    t.xy[ssel, i] = data[hsel, ti]

            # special case of rotor diameter and hub height:
            if v in [FV.D, FV.H]:
                for ti in np.where(tsel)[1]:
                    t = algo.farm.turbines[ti]
                    x = np.zeros(algo.n_states, dtype=FC.DTYPE)
                    if v == FV.D:
                        x[:] = t.D
                        t.D = x
                    else:
                        x[:] = t.H
                        t.H = x
                    i0 = fdata.states_i0()
                    hsel = tsel[:, ti]
                    ssel = i0 + np.where(hsel)[0]
                    x[ssel] = data[hsel, ti]

            fdata[v][tsel] = data[tsel]

        return {v: fdata[v] for v in self.vars}
