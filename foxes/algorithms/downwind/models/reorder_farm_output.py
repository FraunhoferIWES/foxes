import numpy as np

from foxes.core import FarmDataModel
import foxes.variables as FV


class ReorderFarmOutput(FarmDataModel):
    """
    Reorders final results to state-turbine dimensions

    Attributes
    ----------
    outputs: list of str, optional
        The output variables, or None for defaults

    :group: algorithms.downwind.models

    """

    def __init__(self, outputs):
        """
        Constructor

        Parameters
        ----------
        outputs: list of str, optional
            The output variables, or None for defaults

        """
        super().__init__(pre_rotor=False)
        self.outputs = outputs

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
        return self.outputs if self.outputs is not None else algo.farm_vars

    def calculate(self, algo, mdata, fdata):
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

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        ssel = fdata[FV.ORDER_SSEL]
        order_inv = fdata[FV.ORDER_INV]

        out = {}
        for v in self.output_farm_vars(algo):
            if v != FV.ORDER and np.any(fdata[v] != fdata[v][0, 0, None, None]):
                out[v] = fdata[v][ssel, order_inv]
            else:
                out[v] = fdata[v]

        return out
