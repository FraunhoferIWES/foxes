import numpy as np

import foxes.variables as FV
from foxes.core import FarmDataModel, Data


class FarmWakesCalculation(FarmDataModel):
    """
    This model calculates wakes effects on farm data.

    :group: algorithms.downwind.models

    """

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
        # prepare:
        n_turbines = mdata.n_turbines

        # generate all wake evaluation points
        # and all wake deltas, both storing as
        # (n_states, n_order, n_rpoints)
        wpoints = {}
        wdeltas = {}
        for wname, wmodel in algo.wake_models.items():
            pwake = algo.partial_wakes[wname]
            if pwake.name not in wpoints:
                wpoints[pwake.name] = pwake.get_wake_points(
                    algo, mdata, fdata)
            wdeltas[wname] = pwake.new_wake_deltas(
                algo, mdata, fdata, wmodel, wpoints[pwake.name])

        def _get_wdata(wname, s):
            tdata = tdata_all.get_slice(s)
            wdelta = {v: d[s] for v, d in wdeltas[wname].items()}
            return tdata, wdelta

        def _evaluate(algo, mdata, fdata, wdeltas, oi, wmodel, pwake):
            pwake.evaluate_results(
                algo, mdata, fdata, wdeltas, wmodel, oi)
            res = algo.farm_controller.calculate(
                algo, mdata, fdata, pre_rotor=False, downwind_index=oi)
            fdata.update(res)

        for wname, wmodel in algo.wake_models.items():
            pwake = algo.partial_wakes[wname]
            tdata_all = Data.from_tpoints(rpoints=wpoints[pwake.name])

            # downwind:
            if wmodel.affects_downwind:
                for oi in range(n_turbines):
                    if oi > 0:
                        wdelta = {v: d[:, oi] for v, d in wdeltas[wname].items()}
                        _evaluate(algo, mdata, fdata, wdelta, oi, wmodel, pwake)

                    if oi < n_turbines - 1:
                        tdata, wdelta = _get_wdata(wname, np.s_[:, oi+1:])
                        pwake.contribute(algo, mdata, fdata, 
                                         tdata, oi, wdelta, wmodel)
                
            # upwind:
            else:
                for oi in range(n_turbines-1, -1, -1):
                    if oi < n_turbines - 1:
                        wdelta = {v: d[:, oi] for v, d in wdeltas[wname].items()}
                        _evaluate(algo, mdata, fdata, wdelta, oi, wmodel, pwake)

                    if oi > 0:
                        tdata, wdelta = _get_wdata(wname, np.s_[:, :oi])
                        pwake.contribute(algo, mdata, fdata, 
                                         tdata, oi, wdelta, wmodel)

        return {v: fdata[v] for v in self.output_farm_vars(algo)}
