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
        torder = fdata[FV.ORDER]
        n_states = mdata.n_states
        n_turbines = mdata.n_turbines
        ssel = np.zeros_like(torder)
        ssel[:] = np.arange(n_states)[:, None]

        # generate all wake evaluation points
        # and all wake deltas, both storing as
        # (n_states, n_order, n_rpoints)
        wpoints = {}
        wdeltas = {}
        for wname, wmodel in algo.wake_models.items():
            pwake = algo.partial_wakes[wname]
            if pwake.name not in wpoints:
                points = pwake.get_wake_points(algo, mdata, fdata)
                wpoints[pwake.name] = points[ssel,  torder] # reorder turbines
            wdeltas[wname] = pwake.new_wake_deltas(algo, mdata, fdata, wmodel, 
                                                   wpoints[pwake.name])

        def _get_wdata(wname, pwake, s):
            points = wpoints[pwake.name][:, s]
            n_targets, n_rpoints = points.shape[1:3]
            n_points = n_targets * n_rpoints

            pdata = Data.from_points(points=points.reshape(n_states, n_points, 3))

            wdelta = {v: d[:, s].reshape(n_states, n_points) 
                        for v, d in wdeltas[wname].items()}
                        
            return pdata, wdelta

        def _evaluate(algo, mdata, fdata, wdeltas, o, wmodel, pwake):
            pwake.evaluate_results(algo, mdata, fdata, wdeltas, 
                                   wmodel, states_turbine=o)

            trbs = np.zeros((n_states, algo.n_turbines), dtype=bool)
            trbs[np.arange(n_states), o] = True

            res = algo.farm_controller.calculate(
                algo, mdata, fdata, pre_rotor=False, st_sel=trbs
            )
            fdata.update(res)

        for oi in range(n_turbines):
            for wname, wmodel in algo.wake_models.items():
                pwake = algo.partial_wakes[wname]

                # downwind:
                if wmodel.effects_downwind:
                    o = torder[:, oi]

                    if oi > 0:
                        wdelta = {v: d[:, oi] for v, d in wdeltas[wname].items()}
                        _evaluate(algo, mdata, fdata, wdelta, o, wmodel, pwake)

                    if oi < n_turbines - 1:
                        pdata, wdelta = _get_wdata(wname, pwake, np.s_[oi+1:])
                        pwake.contribute_to_wake_deltas(algo, mdata, fdata, 
                                                        pdata, o, wdelta, wmodel)
                
                # upwind:
                else:
                    oj = n_turbines - oi - 1
                    o = torder[:, oj]

                    if oj < n_turbines - 1:
                        wdelta = {v: d[:, oj] for v, d in wdeltas[wname].items()}
                        _evaluate(algo, mdata, fdata, wdelta, o, wmodel, pwake)

                    if oj > 0:
                        pdata, wdelta = _get_wdata(wname, pwake, np.s_[:oj])
                        pwake.contribute_to_wake_deltas(algo, mdata, fdata, 
                                                        pdata, o, wdelta, wmodel)

        return {v: fdata[v] for v in self.output_farm_vars(algo)}
