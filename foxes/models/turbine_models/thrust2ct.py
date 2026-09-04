from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
from typing import TYPE_CHECKING

from foxes.core import TurbineModel
import foxes.variables as FV

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData


class Thrust2Ct(TurbineModel):
    """
    Calculates ct from thrust force data.
    """

    def __init__(self, thrust_var: str = FV.T, var_ws_ct: str = FV.REWS2) -> None:
        """
        Parameters
        ----------
        thrust_var
            Name of the thrust variable
        var_ws_ct
            The wind speed variable for ct lookup
        """
        super().__init__()
        self.thrust_var = thrust_var
        self.WSCT = var_ws_ct

    def __repr__(self) -> str:
        a = f"thrust_var={self.thrust_var}, var_ws_ct={self.WSCT}"
        return f"{type(self).__name__}({a})"

    def output_farm_vars(self, algo: Algorithm) -> list[str]:
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo
            The calculation algorithm

        Returns
        -------
        output_vars
            The output variable names

        """
        return [FV.CT]

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        st_sel: slice | np.ndarray = slice(None),
    ) -> dict[str, np.ndarray]:
        """
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        st_sel: slice or array of bool
            The state-turbine selection,
            for shape: (n_states, n_turbines)

        Returns
        -------
        results
            The resulting data, keys: output variable str.
            Values

        """
        self.ensure_output_vars(algo, fdata)
        ct = fdata[FV.CT]

        T = fdata[self.thrust_var][st_sel]
        rho = fdata[FV.RHO][st_sel]
        A = np.pi * (fdata[FV.D][st_sel] / 2) ** 2
        ws = fdata[self.WSCT][st_sel]

        ct[st_sel] = 2 * T / (rho * A * ws**2)

        return {FV.CT: ct}
