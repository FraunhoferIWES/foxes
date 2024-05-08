import numpy as np

from foxes.core import TurbineModel


class Calculator(TurbineModel):
    """
    Calculates variables based on given functions.

    Attributes
    ----------
    in_vars: list of str
        The input farm vairables
    out_vars: list of str
        The output variables
    func: Function
        The function: f(in0, in1, ..., stsel) -> (out0, ou1, ...)
        where inX and outY are numpy.ndarrays and
        st_sel is the state-turbine selection slice or array.
        All arrays have shape (n_states, n_turbines).

    :group: models.turbine_models

    """

    def __init__(self, in_vars, out_vars, func, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        in_vars: list of str
            The input farm vairables
        out_vars: list of str
            The output variables
        func: Function
            The function: f(in0, in1, ..., stsel) -> (out0, ou1, ...)
            where inX and outY are numpy.ndarrays and
            st_sel is the state-turbine selection slice or array.
            All arrays have shape (n_states, n_turbines).
        kwargs: dict, optional
            Additional arguments for TurbineModel

        """
        super().__init__(**kwargs)
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.func = func

    def __repr__(self):
        a = f"{self.in_vars}, {self.out_vars}"
        return f"{type(self).__name__}({a})"

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
        return self.out_vars

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
        ins = [fdata[v] if v in fdata else mdata[v] for v in self.in_vars]
        outs = self.func(*ins, st_sel=st_sel)

        return {v: outs[vi] for vi, v in enumerate(self.out_vars)}
