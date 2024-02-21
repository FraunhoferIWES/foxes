import numpy as np

from foxes.core import PartialWakesModel, Data
import foxes.constants as FC


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
        rpoints = algo.rotor_model.from_data_or_store(FC.RPOINTS, algo, mdata)
        return rpoints

    def new_wake_deltas(self, algo, mdata, fdata, wmodel):
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
        wmodel: foxes.core.WakeModel
            The wake model

        Returns
        -------
        wake_deltas: dict
            Keys: Variable name str, values: any
        pdata: foxes.core.Data
            The evaluation point data

        """
        points = self.get_wake_points(algo, mdata, fdata)
        n_states, n_turbines, n_rpoints, __ = points.shape
        points = points.reshape(n_states, n_turbines*n_rpoints, 3)
        pdata = Data.from_points(points=points)

        self.N_RPTS = self.var("N_RPTS")
        mdata[self.N_RPTS] = n_rpoints

        wake_deltas = {}
        wmodel.init_wake_deltas(algo, mdata, fdata, pdata, wake_deltas)

        return wake_deltas, pdata

    def contribute_to_wake_deltas(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        states_source_turbine,
        wake_deltas,
        wmodel,  
        wtargets,
    ):
        """
        Modifies wake deltas by contributions from the
        specified wake source turbines.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        pdata: foxes.core.Data
            The evaluation point data
        states_source_turbine: numpy.ndarray of int
            For each state, one turbine index corresponding
            to the wake causing turbine. Shape: (n_states,)
        wake_deltas: Any
            The wake deltas object created by the
            `new_wake_deltas` function
        wmodel: foxes.core.WakeModel
            The wake model
        wtargets: numpy.ndarray
            Boolean flags for active turbines,
            shape: (n_states, n_turbines)

        """
        n_states = fdata.n_states
        n_turbines = fdata.n_turbines
        n_rpoints = mdata[self.N_RPTS]

        points = pdata[FC.POINTS].reshape(n_states, n_turbines, n_rpoints, 3)[wtargets]
        n_targets = int(points.shape[0]/n_states)
        points = points.reshape(n_states, n_targets*n_rpoints, 3)
        hpdata = Data.from_points(points=points)

        wcoos = algo.wake_frame.get_wake_coos(
            algo, mdata, fdata, hpdata, states_source_turbine
        )

        wdel = {}
        for v, d in wake_deltas.items():
            wdel[v] = d.reshape(n_states, n_turbines, n_rpoints)[wtargets]
            wdel[v] = wdel[v].reshape(n_states, n_targets*n_rpoints)

        wmodel.contribute_to_wake_deltas(
            algo, mdata, fdata, hpdata, states_source_turbine, 
            wcoos, wdel
        )
        
        for v, d in wdel.items():
            wake_deltas[v] = wake_deltas[v].reshape(n_states, n_turbines, n_rpoints)
            wake_deltas[v][wtargets] = d.reshape(n_states*n_targets, n_rpoints)
            wake_deltas[v] = wake_deltas[v].reshape(n_states, n_turbines*n_rpoints)

    def evaluate_results(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        wake_deltas,
        wmodel,
        wtargets,
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
        wtargets: numpy.ndarray
            Boolean flags for active turbines,
            shape: (n_states, n_turbines)
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
            wdel[v] = d.reshape(n_states, n_turbines, n_rpoints)[st_sel]
        for w in self.wake_models:
            w.finalize_wake_deltas(algo, mdata, fdata, pdata, wres, wdel)

        for v in wres.keys():
            if v in wake_deltas:
                wres[v] += wdel[v]
                if amb_res_in:
                    amb_res[v][st_sel] = wres[v]
            wres[v] = wres[v][:, None]

        algo.rotor_model.eval_rpoint_results(
            algo, mdata, fdata, wres, weights, states_turbine=states_turbine
        )
