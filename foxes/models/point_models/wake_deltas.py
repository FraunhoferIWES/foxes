from foxes.core import PointDataModel
import foxes.variables as FV


class WakeDeltas(PointDataModel):
    """
    This point model simply subtracts ambient results
    from waked results.

    Parameters
    ----------
    vars : list of str
        The variables
    normalize : bool
        Divide resulting deltas by ambient values

    Attributes
    ----------
    vars : list of str
        The variables
    normalize : bool
        Divide resulting deltas by ambient values

    """

    def __init__(self, vars, normalize=False):
        super().__init__()
        self.vars = vars
        self.normalize = normalize

    def output_point_vars(self, algo):
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
        return [f"DELTA_{v}" for v in self.vars]

    def calculate(self, algo, mdata, fdata, pdata):
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
        pdata : foxes.core.Data
            The point data

        Returns
        -------
        results : dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_points)

        """

        out = {f"DELTA_{v}": pdata[v] - pdata[FV.var2amb[v]] for v in self.vars}

        if self.normalize:
            for v in self.vars:
                out[v] /= pdata[FV.var2amb[v]]

        return out
