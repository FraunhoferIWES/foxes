from .model import Model


class GroundModel(Model):
    """
    Base class for ground models.
    """

    def new_farm_wake_deltas(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        wmodel,
        pwake,
    ):
        """
        Creates new initial wake deltas, filled
        with zeros.

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
        wmodel: foxes.core.WakeModel
            The wake model
        pwake: foxes.core.PartialWakesModel
            The partial wakes model

        Returns
        -------
        wake_deltas: dict
            Key: variable name, value: The zero filled
            wake deltas, shape: (n_states, n_turbines, n_tpoints, ...)

        """
        return pwake.new_wake_deltas(algo, mdata, fdata, tdata, wmodel)

    def contribute_to_farm_wakes(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        wake_deltas,
        wmodel,
        pwake,
    ):
        """
        Modifies wake deltas at target points by
        contributions from the specified wake source turbines.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data
        downwind_index: int
            The index of the wake causing turbine
            in the downwnd order
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints, ...)
        wmodel: foxes.core.WakeModel
            The wake model
        pwake: foxes.core.PartialWakesModel
            The partial wakes model

        """
        pwake.contribute(algo, mdata, fdata, tdata, downwind_index, wake_deltas, wmodel)

    def finalize_farm_wakes(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        amb_res,
        rpoint_weights,
        wake_deltas,
        wmodel,
        downwind_index,
        pwake,
    ):
        """
        Updates the wake_deltas at the selected target
        downwind index.

        Modifies wake_deltas on the fly.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.Data
            The target point data
        amb_res: dict
            The ambient results at the target points
            of all rotors. Key: variable name, value
            np.ndarray of shape:
            (n_states, n_turbines, n_rotor_points)
        rpoint_weights: numpy.ndarray
            The rotor point weights, shape: (n_rotor_points,)
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: np.ndarray of shape
            (n_states, n_turbines, n_tpoints)
        wmodel: foxes.core.WakeModel
            The wake model
        downwind_index: int
            The index in the downwind order

        Returns
        -------
        final_wake_deltas: dict
            The final wake deltas at the selected downwind
            turbines. Key: variable name, value: np.ndarray
            of shape (n_states, n_rotor_points)

        """
        return pwake.finalize_wakes(
            algo,
            mdata,
            fdata,
            tdata,
            amb_res,
            rpoint_weights,
            wake_deltas,
            wmodel,
            downwind_index,
        )

    def new_point_wake_deltas(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        wmodel,
    ):
        """
        Creates new empty wake delta arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data
        wmodel: foxes.core.WakeModel
            The wake model

        Returns
        -------
        wake_deltas: dict
            Key: variable name, value: The zero filled
            wake deltas, shape: (n_states, n_targets, n_tpoints, ...)

        """
        return wmodel.new_wake_deltas(algo, mdata, fdata, tdata)

    def contribute_to_point_wakes(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        wake_deltas,
        wmodel,
    ):
        """
        Modifies wake deltas at target points by
        contributions from the specified wake source turbines.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data
        downwind_index: int
            The index of the wake causing turbine
            in the downwnd order
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints, ...)
        wmodel: foxes.core.WakeModel
            The wake model

        """
        wcoos = algo.wake_frame.get_wake_coos(algo, mdata, fdata, tdata, downwind_index)
        wmodel.contribute(algo, mdata, fdata, tdata, downwind_index, wcoos, wake_deltas)

    def finalize_point_wakes(
        self,
        algo,
        mdata,
        fdata,
        amb_results,
        wake_deltas,
        wmodel,
    ):
        """
        Finalize the wake calculation.

        Modifies wake_deltas on the fly.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        amb_results: dict
            The ambient results, key: variable name str,
            values: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints)
        wake_deltas: dict
            The wake deltas object at the selected target
            turbines. Key: variable str, value: numpy.ndarray
            with shape (n_states, n_targets, n_tpoints)

        """
        wmodel.finalize_wake_deltas(algo, mdata, fdata, amb_results, wake_deltas)
