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
        Get the wake calculation points, and their
        weights.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data

        Returns
        -------
        rpoints: numpy.ndarray
            The wake calculation points, shape:
            (n_states, n_turbines, n_tpoints, 3)
        rweights: numpy.ndarray
            The target point weights, shape: (n_tpoints,)

        """
        return (
            algo.get_from_chunk_store(FC.ROTOR_POINTS, mdata=mdata),
            algo.get_from_chunk_store(FC.ROTOR_WEIGHTS, mdata=mdata),
        )

    def map_rotor_results(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        variable,
        rotor_res,
        rotor_weights,
    ):
        """
        Map ambient rotor point results onto target points.

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
        variable: str
            The variable name to map
        rotor_res: numpy.ndarray
            The results at rotor points, shape:
            (n_states, n_turbines, n_rotor_points)
        rotor_weights: numpy.ndarray
            The rotor point weights, shape: (n_rotor_points,)

        Returns
        -------
        res: numpy.ndarray
            The mapped results at target points, shape:
            (n_states, n_targets, n_tpoints)

        """
        return rotor_res

    def finalize_wakes(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        rpoint_weights,
        wake_deltas,
        wmodel,
        downwind_index,
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
        wdel = {
            v: d[:, downwind_index, None].copy() if d.shape[1] > 1 else d[:, 0, None]
            for v, d in wake_deltas.items()
        }
        wmodel.finalize_wake_deltas(algo, mdata, fdata, tdata, wdel)

        return {v: d[:, 0] for v, d in wdel.items()}
