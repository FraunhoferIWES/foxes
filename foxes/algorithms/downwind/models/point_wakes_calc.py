import foxes.variables as FV
import foxes.constants as FC
from foxes.core import PointDataModel


class PointWakesCalculation(PointDataModel):
    """
    This model calculates wake effects at points of interest.

    Parameters
    ----------
    point_vars : list of str
        The variables of interest
    emodels : foxes.core.PointDataModelList
        The extra evaluation models
    emodels_cpars : list of dict
        The calculation parameters for extra models

    Attributes
    ----------
    point_vars : list of str
        The variables of interest
    emodels : foxes.core.PointDataModelList
        The extra evaluation models
    emodels_cpars : list of dict
        The calculation parameters for extra models

    """

    def __init__(self, point_vars, emodels, emodels_cpars):
        super().__init__()
        self.pvars = point_vars
        self.emodels = emodels
        self.emodels_cpars = emodels_cpars

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
        if self.pvars is None:
            self.pvars = algo.states.output_point_vars(algo)
        return self.pvars

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
        torder = fdata[FV.ORDER].astype(FC.ITYPE)
        n_order = torder.shape[1]
        points = pdata[FV.POINTS]

        wdeltas = {}
        for w in algo.wake_models:
            w.init_wake_deltas(algo, mdata, fdata, pdata.n_points, wdeltas)

        for oi in range(n_order):

            o = torder[:, oi]
            wcoos = algo.wake_frame.get_wake_coos(algo, mdata, fdata, o, points)

            for w in algo.wake_models:
                w.contribute_to_wake_deltas(algo, mdata, fdata, o, wcoos, wdeltas)

        amb_res = {v: pdata[FV.var2amb[v]] for v in wdeltas}
        for w in algo.wake_models:
            w.finalize_wake_deltas(algo, mdata, fdata, amb_res, wdeltas)

        for v in self.pvars:
            if v in wdeltas:
                pdata[v] = amb_res[v] + wdeltas[v]

        self.emodels.calculate(algo, mdata, fdata, pdata, self.emodels_cpars)

        return {v: pdata[v] for v in self.output_point_vars(algo)}
