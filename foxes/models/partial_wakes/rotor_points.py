import numpy as np

from foxes.core import PartialWakesModel
import foxes.variables as FV


class RotorPoints(PartialWakesModel):
    """
    Partial wakes calculation directly by the
    rotor model.

    Parameters
    ----------
    wake_models : list of foxes.core.WakeModel, optional
        The wake models, default are the ones from the algorithm
    wake_frame : foxes.core.WakeFrame, optional
        The wake frame, default is the one from the algorithm

    """

    def __init__(self, wake_models=None, wake_frame=None):
        super().__init__(wake_models, wake_frame)

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        verbosity : int
            The verbosity level

        """
        if not algo.rotor_model.initialized:
            algo.rotor_model.initialize(algo, verbosity=verbosity)

        self.WPOINTS = self.var("WPOINTS")

        super().initialize(algo, verbosity=verbosity)

    def get_wake_points(self, algo, mdata, fdata):
        """
        Get the wake calculation points.

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
        rpoints : numpy.ndarray
            All rotor points, shape: (n_states, n_points, 3)

        """
        rpoints = self.get_data(FV.RPOINTS, mdata)
        n_states, n_turbines, n_rpoints, __ = rpoints.shape
        return rpoints.reshape(n_states, n_turbines * n_rpoints, 3)

    def new_wake_deltas(self, algo, mdata, fdata):
        """
        Creates new initial wake deltas, filled
        with zeros.

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
        wake_deltas : dict
            Keys: Variable name str, values: any

        """
        mdata[self.WPOINTS] = self.get_wake_points(algo, mdata, fdata)
        n_points = mdata[self.WPOINTS].shape[1]

        wake_deltas = {}
        for w in self.wake_models:
            w.init_wake_deltas(algo, mdata, fdata, n_points, wake_deltas)

        return wake_deltas

    def contribute_to_wake_deltas(
        self, algo, mdata, fdata, states_source_turbine, wake_deltas
    ):
        """
        Modifies wake deltas by contributions from the
        specified wake source turbines.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        states_source_turbine : numpy.ndarray of int
            For each state, one turbine index corresponding
            to the wake causing turbine. Shape: (n_states,)
        wake_deltas : Any
            The wake deltas object created by the
            `new_wake_deltas` function

        """
        points = mdata[self.WPOINTS]
        wcoos = self.wake_frame.get_wake_coos(
            algo, mdata, fdata, states_source_turbine, points
        )

        for w in self.wake_models:
            w.contribute_to_wake_deltas(
                algo, mdata, fdata, states_source_turbine, wcoos, wake_deltas
            )

    def evaluate_results(
        self, algo, mdata, fdata, wake_deltas, states_turbine, update_amb_res=False
    ):
        """
        Updates the farm data according to the wake
        deltas.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
            Modified in-place by this function
        wake_deltas : Any
            The wake deltas object, created by the
            `new_wake_deltas` function and filled
            by `contribute_to_wake_deltas`
        states_turbine : numpy.ndarray of int
            For each state, the index of one turbine
            for which to evaluate the wake deltas.
            Shape: (n_states,)
        update_amb_res : bool
            Flag for updating ambient results

        """
        weights = self.get_data(FV.RWEIGHTS, mdata)
        amb_res = self.get_data(FV.AMB_RPOINT_RESULTS, mdata)
        rpoints = self.get_data(FV.RPOINTS, mdata)
        n_states, n_turbines, n_rpoints, __ = rpoints.shape

        wres = {}
        st_sel = (np.arange(n_states), states_turbine)
        for v, ares in amb_res.items():
            wres[v] = ares.reshape(n_states, n_turbines, n_rpoints)[st_sel]
        del amb_res

        wdel = {}
        for v, d in wake_deltas.items():
            wdel[v] = d.reshape(n_states, n_turbines, n_rpoints)[st_sel]
        for w in self.wake_models:
            w.finalize_wake_deltas(algo, mdata, fdata, wres, wdel)

        for v in wres.keys():
            if v in wake_deltas:
                wres[v] += wdel[v]
                if update_amb_res:
                    mdata[FV.AMB_RPOINT_RESULTS][v][st_sel] = wres[v]
            wres[v] = wres[v][:, None]

        algo.rotor_model.eval_rpoint_results(
            algo, mdata, fdata, wres, weights, states_turbine=states_turbine
        )
