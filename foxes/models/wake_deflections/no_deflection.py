
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

    def calc_deflection(
        self,
        algo, 
        mdata,
        fdata, 
        tdata, 
        downwind_index, 
        wframe, 
        coos,
    ):
        """
        Calculates the wake deflection.

        This function optionally adds FC.WDEFL_ROT_ANGLE or
        FC.WDEFL_DWS_FACTOR to the tdata.

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
        coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)

        Returns
        -------
        coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)

        """
        return coos
    
    def get_yaw_alpha_seq(
        self, 
        algo, 
        mdata, 
        fdata, 
        tdata, 
        downwind_index,
        x,
    ):
        """ 
        Computes sequential wind vector rotation angles.

        Wind vector rotation angles are computed at the 
        current trace points due to a yawed rotor
        for sequential runs.

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
        x: numpy.ndarray
            The distance from the wake causing rotor
            for the first n_times subsequent time steps,
            shape: (n_times,)
        
        Returns
        -------
        alpha: numpy.ndarray
            The delta WD result at the x locations,
            shape: (n_times,)
        
        """
        return None
    