import foxes.variables as FV
import foxes.constants as FC
from foxes.core import PointDataModel


class PointWakesCalculation(PointDataModel):
    """
    This model calculates wake effects at points of interest.

    Attributes
    ----------
    point_vars: list of str
        The variables of interest
    emodels: foxes.core.PointDataModelList
        The extra evaluation models
    emodels_cpars: list of dict
        The calculation parameters for extra models
    wake_models: list of foxes.core.WakeModel
        The wake models, default: from algo

    :group: algorithms.downwind.models

    """

    def __init__(
        self, point_vars=None, emodels=None, emodels_cpars=None, wake_models=None
    ):
        """
        Constructor.

        Parameters
        ----------
        point_vars: list of str, optional
            The variables of interest
        emodels: foxes.core.PointDataModelList, optional
            The extra evaluation models
        emodels_cpars: list of dict, optional
            The calculation parameters for extra models
        wake_models: list of foxes.core.WakeModel, optional
            The wake models, default: from algo

        """
        super().__init__()
        self._pvars = point_vars
        self.emodels = emodels
        self.emodels_cpars = emodels_cpars
        self.wake_models = wake_models

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return [self.emodels] if self.emodels is not None else []

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
        super().initialize(algo, verbosity)
        self.pvars = (
            algo.states.output_point_vars(algo) if self._pvars is None else self._pvars
        )

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
        return self.pvars

    def contribute_to_wake_deltas(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        states_source_turbine,
        wmodels,
        wdeltas,
    ):
        """
        Contribute to wake deltas from source turbines

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
        states_source_turbine: numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        wmodels: list of foxes.core.WakeModel
            The wake models
        wdeltas: dict
            The wake deltas, are being modified ob the fly.
            Key: Variable name str, for which the
            wake delta applies, values: numpy.ndarray with
            shape (n_states, n_points, ...)

        """
        wcoos = algo.wake_frame.get_wake_coos(
            algo, mdata, fdata, pdata, states_source_turbine
        )

        for w in wmodels:
            w.contribute_to_wake_deltas(
                algo, mdata, fdata, pdata, states_source_turbine, wcoos, wdeltas
            )

    def calculate(self, algo, mdata, fdata, pdata, states_source_turbine=None):
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
        states_source_turbine: numpy.ndarray, optional
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,).
            Default: include all turbines

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_points)

        """

        torder = fdata[FV.ORDER].astype(FC.ITYPE)
        n_order = torder.shape[1]
        wake_models = algo.wake_models if self.wake_models is None else self.wake_models

        wdeltas = {}
        wmodels = []
        for w in wake_models:
            hdeltas = {}
            w.init_wake_deltas(algo, mdata, fdata, pdata, hdeltas)
            if len(set(self.pvars).intersection(hdeltas.keys())):
                wdeltas.update(hdeltas)
                wmodels.append(w)
            del hdeltas

        if states_source_turbine is None:
            for oi in range(n_order):
                o = torder[:, oi]
                self.contribute_to_wake_deltas(
                    algo, mdata, fdata, pdata, o, wmodels, wdeltas
                )
        else:
            self.contribute_to_wake_deltas(
                algo, mdata, fdata, pdata, states_source_turbine, wmodels, wdeltas
            )

        amb_res = {v: pdata[FV.var2amb[v]] for v in wdeltas if v in FV.var2amb}
        for w in wmodels:
            w.finalize_wake_deltas(algo, mdata, fdata, pdata, amb_res, wdeltas)
        import numpy as np

        for v in self.pvars:
            if v in wdeltas:
                pdata[v] = amb_res[v] + wdeltas[v]

        if self.emodels is not None:
            self.emodels.calculate(algo, mdata, fdata, pdata, self.emodels_cpars)

        return {v: pdata[v] for v in self.pvars}
