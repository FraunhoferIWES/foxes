import numpy as np
from foxes.core import PointDataModel, TData
import foxes.variables as FV
import foxes.constants as FC


class PointWakesCalculation(PointDataModel):
    """
    This model calculates wake effects at points of interest.

    Attributes
    ----------
    pvars: list of str
        The variables of interest
    emodels: foxes.core.PointDataModelList
        The extra evaluation models
    emodels_cpars: list of dict
        The calculation parameters for extra models
    wake_models: list of foxes.core.WakeModel
        The wake models to be used

    :group: algorithms.downwind.models

    """

    def __init__(self, emodels=None, emodels_cpars=None, wake_models=None):
        """
        Constructor.

        Parameters
        ----------
        emodels: foxes.core.PointDataModelList, optional
            The extra evaluation models
        emodels_cpars: list of dict, optional
            The calculation parameters for extra models
        wake_models: list of foxes.core.WakeModel, optional
            Specific wake models to be used

        """
        super().__init__()
        self.pvars = None
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
        self.pvars = algo.states.output_point_vars(algo)

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

    def calculate(self, algo, mdata, fdata, tdata, downwind_index=None):
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
        tdata: foxes.core.Data
            The target point data
        downwind_index: int
            The index in the downwind order of the wake
            causing turbine

        Returns
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints)

        """
        res = {}
        wmodels = (
            algo.wake_models.values() if self.wake_models is None else self.wake_models
        )
        for wmodel in wmodels:
            pwake = algo.partial_wakes[wmodel.name]
            gmodel = algo.ground_models[wmodel.name]

            wdeltas = gmodel.new_point_wake_deltas(algo, mdata, fdata, tdata, wmodel)

            if len(set(self.pvars).intersection(wdeltas.keys())):

                if downwind_index is None:
                    for oi in range(fdata.n_turbines):
                        gmodel.contribute_to_point_wakes(
                            algo, mdata, fdata, tdata, oi, wdeltas, wmodel
                        )
                else:
                    gmodel.contribute_to_point_wakes(
                        algo, mdata, fdata, tdata, downwind_index, wdeltas, wmodel
                    )

                for v in self.pvars:
                    if v not in res and v in tdata:
                        res[v] = tdata[v].copy()

                gmodel.finalize_point_wakes(algo, mdata, fdata, res, wdeltas, wmodel)

                for v in res.keys():
                    if v in wdeltas:
                        res[v] += wdeltas[v]

        for v in res.keys():
            tdata[v] = res[v]

        if self.emodels is not None:
            self.emodels.calculate(algo, mdata, fdata, tdata, self.emodels_cpars)

        return {v: tdata[v] for v in self.pvars}
