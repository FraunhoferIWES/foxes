import numpy as np
from copy import deepcopy

import foxes.variables as FV
from foxes.core import FarmDataModel


class FarmWakesCalculation(FarmDataModel):
    """
    This model calculates wakes effects on farm data.

    :group: algorithms.iterative.models

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
        ovars = deepcopy(algo.rotor_model.output_farm_vars(algo))
        ovars += algo.farm_controller.output_farm_vars(algo)
        return list(dict.fromkeys(ovars))

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        """
        self.pwakes = algo.partial_wakes_model
        super().initialize(algo, verbosity, force=True)

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return []

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
        n_order = torder.shape[1]
        n_states = mdata.n_states

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
            self.pwakes.contribute_to_wake_deltas(algo, mdata, fdata, pdata, o, wdeltas)

        for oi in range(n_order):
            _evaluate(algo, mdata, fdata, pdata, wdeltas, torder[:, oi])

        return {v: fdata[v] for v in self.output_farm_vars(algo)}
