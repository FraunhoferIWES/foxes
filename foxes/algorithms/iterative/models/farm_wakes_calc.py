import numpy as np
from copy import deepcopy

from foxes.core import FarmDataModel
import foxes.constants as FC
import foxes.variables as FV


class FarmWakesCalculation(FarmDataModel):
    """
    This model calculates wakes effects on farm data.

    Attributes
    ----------
    urelax: foxes.algorithms.iterative.models.URelax
        The under-relaxation model

    :group: algorithms.iterative.models

    """

    def __init__(self, urelax=None):
        """
        Constructor.

        Parameters
        ----------
        urelax: foxes.algorithms.iterative.models.URelax, optional
            The under-relaxation model

        """
        super().__init__()
        self.urelax = urelax

    def output_farm_vars(self, algo):
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        output_vars: list of str
            The output variable names

        """
        ovars = algo.rotor_model.output_farm_vars(
            algo
        ) + algo.farm_controller.output_farm_vars(algo)
        return list(dict.fromkeys(ovars))

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            All sub models

        """
        return [] if self.urelax is None else [self.urelax]

    def calculate(self, algo, mdata, fdata):
        """
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

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
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """

        # collect ambient rotor results and weights:
        rotor = algo.rotor_model
        controller = algo.farm_controller
        rwghts = algo.get_from_chunk_store(FC.ROTOR_WEIGHTS, mdata=mdata)
        amb_res = algo.get_from_chunk_store(FC.AMB_ROTOR_RES, mdata=mdata)
        weights = algo.get_from_chunk_store(FC.WEIGHT_RES, mdata=mdata)

        # generate all wake evaluation points
        # (n_states, n_order, n_rpoints)
        pwake2tdata = {}
        pwake2wmodels = {}
        for wname, wmodel in algo.wake_models.items():
            pwake = algo.partial_wakes[wname]
            if pwake.name not in pwake2tdata:
                wmodels = [
                    wm
                    for wn, wm in algo.wake_models.items()
                    if algo.partial_wakes[wn] is pwake
                ]
                pwake2tdata[pwake.name] = pwake.get_initial_tdata(
                    algo, mdata, fdata, amb_res, rwghts, wmodels
                )
                pwake2wmodels[pwake.name] = wmodels

        def _contribute(gmodel, pwake, tdatap, wdeltas, variables, s):
            """Helper function for contribution of wake deltas to wake results"""

            # grab target slice:
            tdata = tdatap.get_slice(variables, s)
            wdelta = {v: d[s] for v, d in wdeltas.items()}

            # reduce to targets within max wake length, if applicable:
            if algo.has_max_wake_length:
                tpts = tdata[FC.TARGETS]
                opts = fdata[FV.TXYH][:, oi]
                tsel = np.all(
                    np.linalg.norm(tpts - opts[:, None, None, :], axis=-1)
                    <= algo.max_wake_length_km * 1e3,
                    axis=(0, 2),
                )
                if not np.any(tsel):
                    return
                wdelta0 = wdelta
                tdata = tdata.get_targets_subset(tsel)
                wdelta = {v: d[:, tsel, ...] for v, d in wdelta0.items()}

            # compute contributions:
            gmodel.contribute_to_farm_wakes(
                algo, mdata, fdata, tdata, oi, wdelta, wmodel, pwake
            )

            # restore full data, if applicable:
            if algo.has_max_wake_length:
                for v in wdelta0.keys():
                    wdelta0[v][:, tsel, ...] = wdelta[v]

        wake_res = deepcopy(amb_res)
        n_turbines = mdata.n_turbines
        for wname, wmodel in algo.wake_models.items():
            pwake = algo.partial_wakes[wname]
            gmodel = algo.ground_models[wname]
            tdatap = pwake2tdata[pwake.name]
            wdeltas = pwake.new_wake_deltas(algo, mdata, fdata, tdatap, wmodel)

            for oi in range(n_turbines):
                if oi > 0:
                    _contribute(
                        gmodel,
                        pwake,
                        tdatap,
                        wdeltas,
                        [FC.STATE, FC.TARGET],
                        np.s_[:, :oi],
                    )

                if oi < n_turbines - 1:
                    _contribute(
                        gmodel,
                        pwake,
                        tdatap,
                        wdeltas,
                        [FC.STATE, FC.TARGET],
                        np.s_[:, oi + 1 :],
                    )

            for oi in range(n_turbines):
                wres = gmodel.finalize_farm_wakes(
                    algo,
                    mdata,
                    fdata,
                    tdatap,
                    rwghts,
                    wdeltas,
                    wmodel,
                    oi,
                    pwake,
                )
                for v, d in wres.items():
                    if v in wake_res:
                        wake_res[v][:, oi] += d

                if controller.has_pre_rotor_models:
                    res = controller.calculate(
                        algo, mdata, fdata, pre_rotor=True, downwind_index=oi
                    )
                    if self.urelax is not None:
                        res = self.urelax.calculate(algo, mdata, fdata, res)
                    fdata.update(res)
                    rotor.calculate(
                        algo,
                        mdata,
                        fdata,
                        downwind_index=oi,
                        rpoint_weights=rwghts,
                        store=True,
                    )
                    weights = algo.get_from_chunk_store(FC.WEIGHT_RES, mdata=mdata)
                    wmdls = pwake2wmodels[pwake.name]
                    pwake.update_tdata(
                        algo, mdata, fdata, tdatap, wake_res, rwghts, wmdls, oi
                    )

            del pwake, tdatap, wdeltas

        wake_res[FV.WEIGHT] = weights
        if not controller.has_pre_rotor_models:
            rotor.eval_rpoint_results(
                algo, mdata, fdata, wake_res, rwghts, set_wd=False
            )

        if controller.has_post_rotor_models:
            res = algo.farm_controller.calculate(algo, mdata, fdata, pre_rotor=False)
            if self.urelax is not None:
                res = self.urelax.calculate(algo, mdata, fdata, res)
            fdata.update(res)

        return {v: fdata[v] for v in self.output_farm_vars(algo)}
