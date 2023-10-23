import numpy as np

from foxes.core import TurbineModel, Data
import foxes.variables as FV
import foxes.constants as FC


class RotorCentreCalc(TurbineModel):
    """
    Calculates data at the rotor centre

    Attributes
    ----------
    calc_vars: dict
        The variables that are calculated by the model,
        keys: var names, values: rotor var names

    :group: models.turbine_models

    """

    def __init__(self, calc_vars):
        """
        Constructor.

        Parameters
        ----------
        calc_vars: dict
            The variables that are calculated by the model,
            keys: var names, values: rotor var names

        """
        super().__init__()

        if isinstance(calc_vars, dict):
            self.calc_vars = calc_vars
        else:
            self.calc_vars = {v: v for v in calc_vars}

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
        pvars = list(self.calc_vars.values())
        self._wcalc = algo.PointWakesCalculation(point_vars=pvars)
        super().initialize(algo, verbosity)

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return [self._wcalc]

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
        return list(self.calc_vars.keys())

    def calculate(self, algo, mdata, fdata, st_sel):
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
        st_sel: numpy.ndarray of bool
            The state-turbine selection,
            shape: (n_states, n_turbines)

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        # prepare point data:
        pdata = {FC.POINTS: fdata[FV.TXYH]}
        dims = {FC.POINTS: (FC.STATE, FC.POINT, FC.XYH)}
        for v in self.calc_vars.values():
            pdata[v] = np.zeros_like(pdata[FC.POINTS][:, :, 0])
            dims[v] = (FC.STATE, FC.POINT)
        pdata = Data(
            name=f"{self.name}_pdata",
            data=pdata,
            dims=dims,
            loop_dims=[FC.STATE, FC.POINT],
        )
        del dims

        # run ambient calculation:
        res = algo.states.calculate(algo, mdata, fdata, pdata)
        for v, a in FV.var2amb.items():
            if v in res:
                res[a] = res[v].copy()
        pdata.update(res)

        # run wake calculation:
        res = self._wcalc.calculate(algo, mdata, fdata, pdata)

        # extract results:
        out = {v: fdata[v] for v in self.calc_vars.keys()}
        for v in out.keys():
            out[v][st_sel] = res[self.calc_vars[v]][st_sel]

        return out
