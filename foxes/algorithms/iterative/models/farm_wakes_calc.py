import numpy as np

import foxes.variables as FV
from foxes.core import FarmDataModel


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
        return [self.pwakes] if self.urelax is None else [self.urelax, self.pwakes]

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
        super().initialize(algo, verbosity)

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

            if self.urelax is not None:
                res = self.urelax.calculate(algo, mdata, fdata)
                for v, d in res.items():
                    fdata[v][trbs] = d[trbs]

        wdeltas, pdata = self.pwakes.new_wake_deltas(algo, mdata, fdata)
        for oi in range(n_order):
            o = torder[:, oi]
            self.pwakes.contribute_to_wake_deltas(algo, mdata, fdata, pdata, o, wdeltas)

        for oi in range(n_order):
            _evaluate(algo, mdata, fdata, pdata, wdeltas, torder[:, oi])

        return {v: fdata[v] for v in self.output_farm_vars(algo)}
