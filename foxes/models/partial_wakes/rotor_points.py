from foxes.core import PartialWakesModel

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
        amb_rotor_res,
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
            The wake deltas object of the selected
            turbines in the downwind order, shape:
            (n_states, 1, n_tpoints, ...)
        wmodel: foxes.core.WakeModel
            The wake model
        downwind_index: int
            The index in the downwind order
        amb_rotor_res: dict
            The ambient results at rotor points of
            the selected turbines in the downwind order. 
            Key: variable name, value: numpy.ndarray 
            with shape (n_states, 1, n_rpoints)
        
        Returns
        -------
        rotor_res: dict
            Waked results at rotor points of
            the selected downwind turbines. Key: variable
            name, value: numpy.ndarray with shape
            (n_states, 1, n_rpoints)

        """
        wmodel.finalize_wake_deltas(algo, mdata, fdata, 
                                    amb_rotor_res, wake_deltas)

        wres = {}
        for v in amb_rotor_res.keys():
            if v in wake_deltas:
                wres[v] = amb_rotor_res[v] + wake_deltas[v]
            else:
                wres[v] = amb_rotor_res[v].copy()

        return wres
        