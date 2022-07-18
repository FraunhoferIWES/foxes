import foxes.variables as FV
from foxes.core import FarmDataModel


class SetAmbFarmResults(FarmDataModel):
    """
    This model copies farm data results to ambient results.

    Parameters
    ----------
    vars_to_amb : list of str, optional
        The variables to be copied, or `None` for all

    Attributes
    ----------
    vars : list of str
        The variables to be copied, or `None` for all

    """

    def __init__(self, vars_to_amb=None):
        super().__init__()
        self.vars = vars_to_amb

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
        if self.vars is None:
            self.vars = set([v for v in algo.farm_vars if v in FV.var2amb])
            self.vars -= set(algo.rotor_model.output_farm_vars(algo))
        return [FV.var2amb[v] for v in self.vars]

    def calculate(self, algo, mdata, fdata):
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

        Returns
        -------
        results : dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        for v in self.vars:
            fdata[FV.var2amb[v]] = fdata[v].copy()
        return {v: fdata[v] for v in self.output_farm_vars(algo)}
