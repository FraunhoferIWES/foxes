import numpy as np

import foxes.variables as FV
import foxes.constants as FC
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
        return fdata[FV.TXYH][:, :, None], np.ones(1, dtype=FC.DTYPE)
