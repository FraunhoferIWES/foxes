import numpy as np

from foxes.core import PartialWakesModel, Data
import foxes.constants as FC


class RotorPoints(PartialWakesModel):
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

    def evaluate_results(
        self,
        algo,
        mdata,
        fdata,
        wake_deltas,
        wmodel,
        downwind_index,
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
        wake_deltas: dict
            The wake deltas object at the selected downwind
            turbines. Key: variable str, value: numpy.ndarray
            with shape (n_states, n_rpoints, ...)
        wmodel: foxes.core.WakeModel
            The wake model
        downwind_index: int
            The index in the downwind order

        """
        weights = algo.rotor_model.from_data_or_store(FC.RWEIGHTS, algo, mdata)
        amb_res = algo.rotor_model.from_data_or_store(
            FC.AMB_RPOINT_RESULTS, algo, mdata
        )
        wres = {v: a[:, downwind_index] for v, a in amb_res.items()}
        del amb_res

        wmodel.finalize_wake_deltas(algo, mdata, fdata, wres, wake_deltas)

        for v in wres.keys():
            if v in wake_deltas:
                print("HERE RPOINTS",v,wres[v].shape,wake_deltas[v].shape)
                wres[v] += wake_deltas[v]
                if amb_res_in:
                    amb_res[v][:, downwind_index] = wres[v]
            wres[v] = wres[v][:, None]

        algo.rotor_model.eval_rpoint_results(
            algo, mdata, fdata, wres, weights, 
            downwind_index=downwind_index
        )
