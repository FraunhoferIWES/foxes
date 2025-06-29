import numpy as np

from foxes.core import PartialWakesModel
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC


class PartialSegregated(PartialWakesModel):
    """
    Add the averaged wake effects to the separately
    averaged ambient rotor results

    Attributes
    ----------
    rotor_model: foxes.core.RotorModel
        The rotor model, default is the one from the algorithm
    grotor: foxes.models.rotor_models.GridRotor
        The grid rotor model

    :group: models.partial_wakes

    """

    def __init__(self, rotor_model):
        """
        Constructor.

        Parameters
        ----------
        rotor_model: foxes.core.RotorModel
            The rotor model for wake averaging

        """
        super().__init__()

        self.rotor = rotor_model
        self.YZ = self.var("YZ")
        self.W = self.var(FV.WEIGHT)

    def __repr__(self):
        return f"{type(self).__name__}(rotor_model={self.rotor.name})"

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return super().sub_models() + [self.rotor]

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
            self.rotor.get_rotor_points(algo, mdata, fdata),
            self.rotor.rotor_point_weights(),
        )

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
        n_states = fdata.n_states
        n_rotor_points = len(rpoint_weights)
        gweights = tdata[FC.TWEIGHTS]

        wdel = {v: d[:, downwind_index, None].copy() for v, d in wake_deltas.items()}
        htdata = tdata.get_slice([FC.TURBINE], np.s_[downwind_index])
        wmodel.finalize_wake_deltas(algo, mdata, fdata, htdata, wdel)

        for v in wdel.keys():
            hdel = np.zeros((n_states, n_rotor_points), dtype=config.dtype_double)
            hdel[:] = np.einsum("sp,p->s", wdel[v][:, 0], gweights)[:, None]
            wdel[v] = hdel

        return wdel
