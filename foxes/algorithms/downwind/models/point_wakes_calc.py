import foxes.variables as FV
import foxes.constants as FC
from foxes.core import PointDataModel

class PointWakesCalculation(PointDataModel):
    """
    This model calculates wake effects at points of interest.

    Parameters
    ----------
    point_vars : list of str, optional
        The variables of interest
    emodels : foxes.core.PointDataModelList, optional
        The extra evaluation models
    emodels_cpars : list of dict, optional
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

    def __init__(self, point_vars=None, emodels=None, emodels_cpars=None):
        super().__init__()
        self._pvars = point_vars
        self.emodels = emodels
        self.emodels_cpars = emodels_cpars

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        This includes loading all required data from files. The model
        should return all array type data as part of the idata return
        dictionary (and not store it under self, for memory reasons). This
        data will then be chunked and provided as part of the mdata object
        during calculations.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        idata : dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """
        self.pvars = algo.states.output_point_vars(algo) if self._pvars is None else self._pvars

        idata = super().initialize(algo, verbosity)
        if self.emodels is not None:
            algo.update_idata(self.emodels, idata=idata, verbosity=verbosity)

        return idata

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
        wmodels = []
        for w in algo.wake_models:
            hdeltas = {}
            w.init_wake_deltas(algo, mdata, fdata, pdata.n_points, hdeltas)
            if len(set(self.pvars).intersection(hdeltas.keys())):
                wdeltas.update(hdeltas)
                wmodels.append(w)
            del hdeltas

        for oi in range(n_order):

            o = torder[:, oi]
            wcoos = algo.wake_frame.get_wake_coos(algo, mdata, fdata, o, points)

            for w in wmodels:
                w.contribute_to_wake_deltas(algo, mdata, fdata, o, wcoos, wdeltas)

        amb_res = {v: pdata[FV.var2amb[v]] for v in wdeltas}
        for w in wmodels:
            w.finalize_wake_deltas(algo, mdata, fdata, amb_res, wdeltas)

        for v in self.pvars:
            if v in wdeltas:
                pdata[v] = amb_res[v] + wdeltas[v]

        if self.emodels is not None:
            self.emodels.calculate(algo, mdata, fdata, pdata, self.emodels_cpars)

        return {v: pdata[v] for v in self.pvars}
