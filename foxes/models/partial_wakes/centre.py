import numpy as np

import foxes.variables as FV
import foxes.constants as FC
from foxes.config import config

from .rotor_points import RotorPoints


class PartialCentre(RotorPoints):
    """
    Partial wakes calculated only at the
    rotor centre point.

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
        return fdata[FV.TXYH][:, :, None], np.ones(1, dtype=config.dtype_double)

    def map_rotor_results(self, algo, mdata, fdata, tdata, variable, rotor_res):
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
        
        Returns
        -------
        res: numpy.ndarray
            The mapped results at target points, shape:
            (n_states, n_targets, n_tpoints)

        """
        if rotor_res.shape[2] > 1:
            return np.einsum(
                "str,r->st",
                rotor_res,
                tdata[FC.TWEIGHTS],
            )[:, :, None]
        else:
            return rotor_res