import numpy as np
from copy import deepcopy

import foxes.variables as FV
from foxes.core import FarmDataModel


class FarmWakesCalculation(FarmDataModel):
    """
    This model calculates wakes effects on farm data.
    """

    def output_farm_vars(self, algo):
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        output_vars : list of str
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
        algo : foxes.core.Algorithm
            The calculation algorithm
        verbosity : int
            The verbosity level

        """
        super().initialize(algo, verbosity=verbosity)
        self.pwakes = algo.partial_wakes_model
        if not self.pwakes.initialized:
            if verbosity:
                print(f"{self.name}, linked model '{self.pwakes.name}': Initializing")
            self.pwakes.initialize(algo, verbosity=verbosity)

    def calculate(self, algo, mdata, fdata):
        """ "
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data

        Returns
        -------
        results : dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        torder = fdata[FV.ORDER]
        n_order = torder.shape[1]
        n_states = mdata.n_states

        wdeltas = self.pwakes.new_wake_deltas(algo, mdata, fdata)

        for oi in range(n_order):

            o = torder[:, oi]

            if oi > 0:

                self.pwakes.evaluate_results(
                    algo, mdata, fdata, wdeltas, states_turbine=o
                )

                trbs = np.zeros((n_states, algo.n_turbines), dtype=bool)
                np.put_along_axis(trbs, o[:, None], True, axis=1)

                res = algo.farm_controller.calculate(
                    algo, mdata, fdata, pre_rotor=False, st_sel=trbs
                )
                fdata.update(res)

            if oi < n_order - 1:
                self.pwakes.contribute_to_wake_deltas(algo, mdata, fdata, o, wdeltas)

        return {v: fdata[v] for v in self.output_farm_vars(algo)}
