import numpy as np

import foxes.constants as FC
import foxes.variables as FV

from .rotor_points import RotorPoints


class PartialCentre(RotorPoints):
    """
    Partial wakes calculated only at the 
    rotor centre point.

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
        return fdata[FV.TXYH][:, :, None]

    def evaluate_results(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        wake_deltas,
        wmodel,
        states_turbine,
        amb_res=None,
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
        pdata: foxes.core.Data
            The evaluation point data
        wake_deltas: Any
            The wake deltas object, created by the
            `new_wake_deltas` function and filled
            by `contribute_to_wake_deltas`
        wmodel: foxes.core.WakeModel
            The wake model
        states_turbine: numpy.ndarray of int
            For each state, the index of one turbine
            for which to evaluate the wake deltas.
            Shape: (n_states,)
        amb_res: dict, optional
            Ambient states results. Keys: var str, values:
            numpy.ndarray of shape (n_states, n_points)

        """
        weights = algo.rotor_model.from_data_or_store(FC.RWEIGHTS, algo, mdata)
        rpoints = algo.rotor_model.from_data_or_store(FC.RPOINTS, algo, mdata)
        n_states, n_turbines, n_rpoints, __ = rpoints.shape

        amb_res_in = amb_res is not None
        if not amb_res_in:
            amb_res = algo.rotor_model.from_data_or_store(
                FC.AMB_RPOINT_RESULTS, algo, mdata
            )

        wres = {}
        st_sel = (np.arange(n_states), states_turbine)
        for v, ares in amb_res.items():
            wres[v] = ares.reshape(n_states, n_turbines, n_rpoints)[st_sel]

        
        wdel = {}
        for v, d in wake_deltas.items():
            wdel[v] = d.reshape(n_states, n_turbines, 1)[st_sel]
        print("HERE PWAKE CENTRE", {v: d.shape for v, d in wdel.items()})
        wmodel.finalize_wake_deltas(algo, mdata, fdata, pdata, wres, wdel)

        for v in wres.keys():
            if v in wake_deltas:
                wres[v] += wdel[v]
                if amb_res_in:
                    amb_res[v][st_sel] = wres[v]
            wres[v] = wres[v][:, None]

        algo.rotor_model.eval_rpoint_results(
            algo, mdata, fdata, wres, weights, states_turbine=states_turbine
        )
