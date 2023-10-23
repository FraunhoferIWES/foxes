import foxes.variables as FV
from foxes.core import PointDataModel


class SetAmbPointResults(PointDataModel):
    """
    This model copies point results to ambient results.

    Attributes
    ----------
    pvars: list of str
        The point variables to be treated
    vars: list of str
        The variables to be copied to output

    :group: algorithms.downwind.models

    """

    def __init__(self, point_vars=None, vars_to_amb=None):
        """
        Constructor.

        Parameters
        ----------
        point_vars: list of str, optional
            The point variables to be treated
        vars_to_amb: list of str, optional
            The variables to be copied to output

        """
        super().__init__()
        self._pvars = point_vars
        self._vars = vars_to_amb

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        """
        self.pvars = (
            algo.states.output_point_vars(algo) if self._pvars is None else self._pvars
        )
        self.vars = (
            [v for v in self.pvars if v in FV.var2amb]
            if self._vars is None
            else self._vars
        )
        super().initialize(algo, verbosity)

    def output_point_vars(self, algo):
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
        for v in algo.states.output_point_vars(algo):
            if v not in self.vars and v in FV.var2amb:
                self.vars.append(v)
        return [FV.var2amb[v] for v in self.vars]

    def calculate(self, algo, mdata, fdata, pdata):
        """ "
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        pdata: foxes.core.Data
            The point data

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_points)

        """
        for v in self.vars:
            pdata[FV.var2amb[v]] = pdata[v].copy()
        return {v: pdata[v] for v in self.output_point_vars(algo)}
