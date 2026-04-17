import numpy as np

from foxes.core import PointDataModel
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

    def initialize(self, algo, verbosity=0, force=False):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent
        force: bool
            Overwrite existing data

        """
        super().initialize(algo, verbosity, force)
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
        """
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

        def _contribute(gmodel, tdata, oi, wdeltas, wmodel):
            """Helper function for contribution of wake deltas to wake results"""

            # reduce to targets within max wake length, if applicable:
            if algo.has_max_wake_length:
                tpts = tdata[FC.TARGETS]
                opts = fdata[FV.TXYH][:, oi]
                tsel = np.all(
                    np.linalg.norm(tpts - opts[:, None, None, :], axis=-1)
                    <= algo.max_wake_length_km * 1e3,
                    axis=(0, 2),
                )
                if not np.any(tsel):
                    return
                wdeltas0 = wdeltas
                tdata = tdata.get_targets_subset(tsel)
                wdeltas = {v: d[:, tsel, ...] for v, d in wdeltas0.items()}

            # compute contributions:
            gmodel.contribute_to_point_wakes(
                algo, mdata, fdata, tdata, oi, wdeltas, wmodel
            )

            # restore full data, if applicable:
            if algo.has_max_wake_length:
                for v in wdeltas0.keys():
                    wdeltas0[v][:, tsel, ...] = wdeltas[v]

        wmodels = (
            algo.wake_models.values() if self.wake_models is None else self.wake_models
        )
        pvrs = self.pvars + [FV.UV]
        for wmodel in wmodels:
            gmodel = algo.ground_models[wmodel.name]

            wdeltas = gmodel.new_point_wake_deltas(algo, mdata, fdata, tdata, wmodel)

            if len(set(pvrs).intersection(wdeltas.keys())):
                if downwind_index is None:
                    for oi in range(fdata.n_turbines):
                        _contribute(gmodel, tdata, oi, wdeltas, wmodel)
                else:
                    _contribute(gmodel, tdata, downwind_index, wdeltas, wmodel)

                gmodel.finalize_point_wakes(algo, mdata, fdata, tdata, wdeltas, wmodel)

                for v in tdata.keys():
                    if v in wdeltas:
                        tdata[v] += wdeltas[v]

        if self.emodels is not None:
            self.emodels.calculate(algo, mdata, fdata, tdata, self.emodels_cpars)

        return {v: tdata[v] for v in self.pvars}
