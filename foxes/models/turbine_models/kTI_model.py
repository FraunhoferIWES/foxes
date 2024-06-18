import numpy as np

from foxes.core import TurbineModel
import foxes.variables as FV
import foxes.constants as FC


class kTI(TurbineModel):
    """
    Calculates the wake model parameter `k`
    as a linear function of `TI`.

    Attributes
    ----------
    ti_var: str
        The `TI` variable name
    k_var: str
        The variable name for k

    :group: models.turbine_models

    """

    def __init__(self, kTI=None, kb=None, ti_var=FV.TI, ti_val=None, k_var=FV.K):
        """
        Constructor.

        Parameters
        ----------
        kTI: float, optional
            Uniform value for `kTI`. If not given it
            will be searched in farm data
        kb: float, optional
            Uniform value for `kb`. If not given it
            will be searched in farm data, and zero by default
        ti_var: str
            The `TI` variable name
        ti_val: float, optional
            The uniform value of `TI`. If not given it
            will be searched in farm data
        k_var: str
            The variable name for k

        """
        super().__init__()

        self.ti_var = ti_var
        self.k_var = k_var
        setattr(self, ti_var, ti_val)
        setattr(self, FV.KTI, kTI)
        setattr(self, FV.KB, 0 if kb is None else kb)

    def __repr__(self):
        kti = getattr(self, FV.KTI)
        kb = getattr(self, FV.KB)
        ti = getattr(self, self.ti_var)
        tiv = f", ti_val={ti}" if ti is not None else ""
        a = f"kTI={kti}, kb={kb}, ti_var={self.ti_var}{tiv}, k_var={self.k_var}"
        return f"{type(self).__name__}({a})"

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
        return [self.k_var]

    def calculate(self, algo, mdata, fdata, st_sel):
        """
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        st_sel: slice or numpy.ndarray of bool
            The state-turbine selection,
            for shape: (n_states, n_turbines)

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        kti = self.get_data(
            FV.KTI, FC.STATE_TURBINE, lookup="sf", fdata=fdata, upcast=True
        )[st_sel]
        kb = self.get_data(
            FV.KB, FC.STATE_TURBINE, lookup="sf", fdata=fdata, upcast=True
        )[st_sel]
        ti = self.get_data(
            self.ti_var, FC.STATE_TURBINE, lookup="sf", fdata=fdata, upcast=True
        )[st_sel]

        k = fdata.get(
            self.k_var, np.zeros((fdata.n_states, fdata.n_turbines), dtype=FC.DTYPE)
        )

        k[st_sel] = kti * ti + kb

        return {self.k_var: k}
