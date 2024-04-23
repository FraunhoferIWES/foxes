import numpy as np

from foxes.core import PartialWakesModel
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
        return super().__repr__() + f"[{self.rotor}]"

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
        return self.rotor.get_rotor_points(algo, mdata, fdata)

    def finalize_wakes(
        self,
        algo, 
        mdata, 
        fdata, 
        amb_res, 
        wake_deltas, 
        wmodel, 
        downwind_index
    ):
        """
        Updates the wake_deltas at the selected target
        downwind index.

        Modifies wake_deltas on the fly.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        amb_res: dict
            The ambient results at the target points
            of all rotors. Key: variable name, value
            np.ndarray of shape: 
            (n_states, n_turbines, n_rotor_points)
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
        ares = {v: d[:, downwind_index, None] for v, d in amb_res.items()}

        gweights = self.rotor.rotor_point_weights()
        wdel = {v: np.einsum('sp,p->s', d[:, downwind_index], gweights)[:, None, None]
                for v, d in wake_deltas.items()}
        
        wmodel.finalize_wake_deltas(algo, mdata, fdata, ares, wdel)

        return {v: d[:, 0] for v, d in wdel.items()}
    
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
            The wake deltas object. Key: variable str, 
            value: numpy.ndarray with shape 
            (n_states, n_targets, n_tpoints, ...)
        wmodel: foxes.core.WakeModel
            The wake model
        downwind_index: int
            The index in the downwind order

        """
        rotor = algo.rotor_model
        weights = rotor.from_data_or_store(rotor.RWEIGHTS, algo, mdata)
        amb_res = rotor.from_data_or_store(rotor.AMBRES, algo, mdata)
        wres = {v: a[:, downwind_index, None].copy() for v, a in amb_res.items()}
        del amb_res
        
        wdel = {v: d[:, downwind_index, None] for v, d in wake_deltas.items()}
        wmodel.finalize_wake_deltas(algo, mdata, fdata, wres, wdel)
        
        gweights = self.rotor.rotor_point_weights()
        for v in wres.keys():
            if v in wake_deltas:
                wres[v] += np.einsum('stp,p->st', wdel[v], gweights)[:, None]

        algo.rotor_model.eval_rpoint_results(
            algo, mdata, fdata, wres, weights, 
            downwind_index=downwind_index
        )
        