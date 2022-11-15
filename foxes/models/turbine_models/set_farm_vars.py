import numpy as np

from foxes.core import TurbineModel
import foxes.constants as FC
import foxes.variables as FV


class SetFarmVars(TurbineModel):
    """
    Set farm data variables to given data.

    Parameters
    ----------
    pre_rotor : bool
        Flag for running this model before
        running the rotor model.

    Attributes
    ----------
    vars : list of str
        The variables to be set

    """

    def __init__(self, pre_rotor=False):
        super().__init__(pre_rotor=pre_rotor)

        self.vars = []
        self._vdata = []

    def add_var(self, var, data):
        """
        Add data for a variable.

        Parameters
        ----------
        var : str
            The variable name
        data : numpy.ndarray
            The data, shape: (n_states, n_turbines)

        """
        self.vars.append(var)
        self._vdata.append(data)

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
        return self.vars

    def model_input_data(self, algo):
        """
        The model input data, as needed for the
        calculation.

        This function should specify all data
        that depend on the loop variable (e.g. state),
        or that are intended to be shared between chunks.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        idata : dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """

        idata = super().model_input_data(algo)

        for i, v in enumerate(self.vars):

            data = np.full((algo.n_states, algo.n_turbines), np.nan, dtype=FC.DTYPE)
            data[:] = self._vdata[i]

            idata["data_vars"][self.var(v)] = ((FV.STATE, FV.TURBINE), data)

        return idata

    def calculate(self, algo, mdata, fdata, st_sel):
        """ "
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        st_sel : numpy.ndarray of bool
            The state-turbine selection,
            shape: (n_states, n_turbines)

        Returns
        -------
        results : dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        n_states = fdata.n_states
        n_turbines = fdata.n_turbines
        allt = np.all(st_sel)

        for v in self.vars:

            data = mdata[self.var(v)]
            hsel = ~np.isnan(data)
            hallt = np.all(hsel)

            if allt and hallt:
                fdata[v][:] = data

            else:

                if v not in fdata:
                    fdata[v] = np.full((n_states, n_turbines), np.nan, dtype=FC.DTYPE)

                tsel = st_sel & hsel
                fdata[v][tsel] = data[tsel]

        return {v: fdata[v] for v in self.vars}
