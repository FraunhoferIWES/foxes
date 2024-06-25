import numpy as np
from copy import deepcopy

from foxes.core import FarmDataModel, TData


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
        """ "
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
        weights = rotor.from_data_or_store(rotor.RWEIGHTS, algo, mdata)
        amb_res = rotor.from_data_or_store(rotor.AMBRES, algo, mdata)

        # generate all wake evaluation points
        # (n_states, n_order, n_rpoints)
        pwake2tdata = {}
        for wname, wmodel in algo.wake_models.items():
            pwake = algo.partial_wakes[wname]
            if pwake.name not in pwake2tdata:
                tpoints, tweights = pwake.get_wake_points(algo, mdata, fdata)
                pwake2tdata[pwake.name] = TData.from_tpoints(tpoints, tweights)

        def _get_wdata(tdatap, wdeltas, s):
            """Helper function for wake data extraction"""
            tdata = tdatap.get_slice(s, keep=True)
            wdelta = {v: d[s] for v, d in wdeltas.items()}
            return tdata, wdelta

        wake_res = deepcopy(amb_res)
        n_turbines = mdata.n_turbines
        for wname, wmodel in algo.wake_models.items():
            pwake = algo.partial_wakes[wname]
            gmodel = algo.ground_models[wname]
            tdatap = pwake2tdata[pwake.name]
            wdeltas = pwake.new_wake_deltas(algo, mdata, fdata, tdatap, wmodel)

            for oi in range(n_turbines):

                if oi > 0:
                    tdata, wdelta = _get_wdata(tdatap, wdeltas, np.s_[:, :oi])
                    gmodel.contribute_to_farm_wakes(
                        algo, mdata, fdata, tdata, oi, wdelta, wmodel, pwake
                    )

                if oi < n_turbines - 1:
                    tdata, wdelta = _get_wdata(tdatap, wdeltas, np.s_[:, oi + 1 :])
                    gmodel.contribute_to_farm_wakes(
                        algo, mdata, fdata, tdata, oi, wdelta, wmodel, pwake
                    )

            for oi in range(n_turbines):
                wres = gmodel.finalize_farm_wakes(
                    algo,
                    mdata,
                    fdata,
                    tdatap,
                    amb_res,
                    weights,
                    wdeltas,
                    wmodel,
                    oi,
                    pwake,
                )
                for v, d in wres.items():
                    if v in wake_res:
                        wake_res[v][:, oi] += d

            del pwake, tdatap, wdeltas

        rotor.eval_rpoint_results(algo, mdata, fdata, wake_res, weights)
        res = algo.farm_controller.calculate(algo, mdata, fdata, pre_rotor=False)
        if self.urelax is not None:
            res = self.urelax.calculate(algo, mdata, fdata, res)
        fdata.update(res)

        return {v: fdata[v] for v in self.output_farm_vars(algo)}
