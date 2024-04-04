import numpy as np

from foxes.core import PartialWakesModel, Data
import foxes.constants as FC

from .wake_shaped import WakeShapedPartials

class RotorPoints(WakeShapedPartials):
    """
    Partial wakes calculation directly by the
    rotor model.

    :group: models.partial_wakes

    """

    def get_wake_points(self, algo, mdata, fdata):
        """
        Get the wake calculation points.

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
        rpoints: numpy.ndarray
            All rotor points, shape: (n_states, n_targets, n_rpoints, 3)

        """
        return algo.rotor_model.from_data_or_store(FC.RPOINTS, algo, mdata)

    def n_wpoints(self, algo):
        """
        The number of evaluation points per rotor

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        n: int
            The number of evaluation points per rotor
        
        """
        return algo.rotor_model.n_rotor_points()

    def evaluate_results(
        self,
        algo,
        mdata,
        fdata,
        wake_deltas,
        wmodel,
        downwind_index,
        amb_res=None,
    ):
        """
        Updates the farm data according to the wake
        deltas.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
            Modified in-place by this function
        wake_deltas: Any
            The wake deltas object at the selected downwind
            turbines
        wmodel: foxes.core.WakeModel
            The wake model
        downwind_index: int
            The index in the downwind order
        amb_res: dict, optional
            Ambient states results. Keys: var str, values:
            numpy.ndarray of shape (n_states, n_points)

        """
        weights = algo.rotor_model.from_data_or_store(FC.RWEIGHTS, algo, mdata)
        rpoints = algo.rotor_model.from_data_or_store(FC.RPOINTS, algo, mdata)
        n_states, n_turbines, n_rpoints, __ = rpoints.shape

        amb_res_in = amb_res is not None
        if not amb_res_in:
            amb_res = algo.rotor_model.from_data_or_store(
                FC.AMB_RPOINT_RESULTS, algo, mdata
            )

        wres = {}
        for v, ares in amb_res.items():
            wres[v] = ares.reshape(n_states, n_turbines, n_rpoints)[:, downwind_index]

        wmodel.finalize_wake_deltas(algo, mdata, fdata, wres, wake_deltas)

        for v in wres.keys():
            if v in wake_deltas:
                wres[v] += wake_deltas[v]
                if amb_res_in:
                    amb_res[v][:, downwind_index] = wres[v]
            wres[v] = wres[v][:, None]

        algo.rotor_model.eval_rpoint_results(
            algo, mdata, fdata, wres, weights, 
            downwind_index=downwind_index
        )
