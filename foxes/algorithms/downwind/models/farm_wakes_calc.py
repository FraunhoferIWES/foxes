import numpy as np

import foxes.variables as FV
from foxes.core import FarmDataModel


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
        torder = fdata[FV.ORDER]
        n_states = mdata.n_states
        n_turbines = mdata.n_turbines

        def _evaluate(algo, mdata, fdata, pdata, wdeltas, o, wmodel, pwake):
            pwake.evaluate_results(algo, mdata, fdata, pdata, wdeltas, 
                                   wmodel, states_turbine=o)

            trbs = np.zeros((n_states, algo.n_turbines), dtype=bool)
            trbs[np.arange(n_states), o] = True

            res = algo.farm_controller.calculate(
                algo, mdata, fdata, pre_rotor=False, st_sel=trbs
            )
            fdata.update(res)

        # downwind:
        wmodels = {w: m for w, m in algo.wake_models.items() if m.effects_downwind}
        if len(wmodels):
            wdeltas = {}
            pdata = {}
            wtargets = np.ones((n_states, n_turbines), dtype=bool)
            for oi in range(n_turbines):
                o = torder[:, oi]
                wtargets[np.arange(n_states), o] = False

                for wname, wmodel in wmodels.items():
                    pwake = algo.partial_wakes[wname]

                    if oi > 0:
                        #print("EVAL",oi, wname)
                        _evaluate(algo, mdata, fdata, pdata[wname], 
                                wdeltas[wname], o, wmodel, pwake)
                        """
                        wdel = {}
                        for oj in range(oi):
                            for v, d in wdeltas[wname][oj].items():
                                if v not in wdel:
                                    wdel[v] = []
                                wdel[v].append(d[:, 0, None])
                                wdeltas[wname][oj][v] = d[:, 1:]
                        wdel = {v: np.concatenate(d, axis=1) for v, d in wdel.items()}
                        print("-->", {v: d.shape for v, d in wdel.items()})
                        _evaluate(algo, mdata, fdata, wdel, o, wmodel, pwake)
                        """

                    if oi < n_turbines - 1:

                        #print("WAKES",oi, wname, np.sum(wtargets[0]))

                        if wname not in wdeltas:
                            wdeltas[wname], pdata[wname] = pwake.new_wake_deltas(
                                algo, mdata, fdata, wmodel)

                        pwake.contribute_to_wake_deltas(algo, mdata, fdata, 
                                pdata[wname], o, wdeltas[wname], wmodel, wtargets)
                



        """
        def _evaluate(algo, mdata, fdata, pdata, wdeltas, o):
            self.pwakes.evaluate_results(
                algo, mdata, fdata, pdata, wdeltas, states_turbine=o
            )

            trbs = np.zeros((n_states, algo.n_turbines), dtype=bool)
            np.put_along_axis(trbs, o[:, None], True, axis=1)

            res = algo.farm_controller.calculate(
                algo, mdata, fdata, pre_rotor=False, st_sel=trbs
            )
            fdata.update(res)

        wdeltas, pdata = self.pwakes.new_wake_deltas(algo, mdata, fdata)
        for oi in range(n_order):
            o = torder[:, oi]

            if oi > 0:
                _evaluate(algo, mdata, fdata, pdata, wdeltas, o)

            if oi < n_order - 1:
                self.pwakes.contribute_to_wake_deltas(
                    algo, mdata, fdata, pdata, o, wdeltas
                )

        """

        return {v: fdata[v] for v in self.output_farm_vars(algo)}
