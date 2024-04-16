from foxes.core import PartialWakesModel
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
        rotor = algo.rotor_model
        return rotor.from_data_or_store(rotor.RPOINTS, algo, mdata)

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
            with shape (n_states, n_tpoints, ...)
        wmodel: foxes.core.WakeModel
            The wake model
        downwind_index: int
            The index in the downwind order

        """
        rotor = algo.rotor_model
        weights = rotor.from_data_or_store(rotor.RWEIGHTS, algo, mdata)
        amb_res = rotor.from_data_or_store(rotor.AMBRES, algo, mdata)
        wres = {v: a[:, downwind_index] for v, a in amb_res.items()}
        del amb_res

        wmodel.finalize_wake_deltas(algo, mdata, fdata, wres, wake_deltas)

        for v in wres.keys():
            if v in wake_deltas:
                wres[v] += wake_deltas[v]
            wres[v] = wres[v][:, None]

        algo.rotor_model.eval_rpoint_results(
            algo, mdata, fdata, wres, weights, 
            downwind_index=downwind_index
        )
