
import numpy as np

from foxes.config import config
from foxes.core.wake_deflection import WakeDeflection
from foxes.core.wake_model import WakeK
from foxes.models.wake_models.wind.bastankhah16 import (
    Bastankhah2016Model,
    Bastankhah2016,
)
import foxes.constants as FC
import foxes.variables as FV


class NoDeflection(WakeDeflection):
    """
    Switch of wake deflection

    :group: models.wake_deflections

    """

    def update_coos(
        self,
        algo, 
        mdata,
        fdata, 
        tdata, 
        downwind_index, 
        wframe, 
        wmodel, 
        coos,
    ):
        """
        Updates the wake coordinates

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
            in the downwind order
        wframe: foxes.core.WakeFrame
            The wake frame
        wmodel: foxes.core.WakeModel
            The wake model
        coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)

        """
        return coos
    