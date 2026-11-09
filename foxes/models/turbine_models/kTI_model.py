from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
from typing import TYPE_CHECKING

from foxes.core import TurbineModel
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData


class kTI(TurbineModel):
    """
    Calculates the wake model parameter `k`
    as a linear function of `TI`.
    """

    def __init__(
        self,
        kTI: float | None = None,
        kb: float | None = None,
        ti_var: str = FV.TI,
        ti_val: float | None = None,
        k_var: str = FV.K,
    ) -> None:
        """
        Parameters
        ----------
        kTI
            Uniform value for `kTI`. If not given it
            will be searched in farm data
        kb
            Uniform value for `kb`. If not given it
            will be searched in farm data, and zero by default
        ti_var
            The `TI` variable name
        ti_val
            The uniform value of `TI`. If not given it
            will be searched in farm data
        k_var
            The variable name for k
        """
        super().__init__()

        self.ti_var = ti_var
        self.k_var = k_var
        setattr(self, ti_var, ti_val)
        setattr(self, FV.KTI, kTI)
        setattr(self, FV.KB, 0 if kb is None else kb)

    def __repr__(self) -> str:
        kti = getattr(self, FV.KTI)
        kb = getattr(self, FV.KB)
        ti = getattr(self, self.ti_var)
        tiv = f", ti_val={ti}" if ti is not None else ""
        a = f"kTI={kti}, kb={kb}, ti_var={self.ti_var}{tiv}, k_var={self.k_var}"
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
        return [self.k_var]

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        st_sel: slice | np.ndarray = slice(None),
    ) -> dict[str, np.ndarray]:
        """
        Calculate wake growth coefficients for selected farm entries.

        Parameters
        ----------
        algo
            The calculation algorithm.
        mdata
            The model data.
        fdata
            The farm data; the coefficient field is updated in place.
        st_sel
            A state slice or boolean state-turbine mask with shape
            ``(n_states, n_turbines)``.

        Returns
        -------
        results
            The coefficient array under ``self.k_var`` with dimensions
            ``(FC.STATE, FC.TURBINE)``. Unselected entries are unchanged.

        """
        self.ensure_output_vars(algo, fdata)

        n_states = fdata.n_states
        n_turbines = fdata.n_turbines
        assert n_states is not None and n_turbines is not None
        kti = self.get_data(
            FV.KTI,
            FC.STATE_TURBINE,
            lookup="sf",
            fdata=fdata,
            selection=st_sel,
        )
        kb = self.get_data(
            FV.KB,
            FC.STATE_TURBINE,
            lookup="sf",
            fdata=fdata,
            selection=st_sel,
        )
        ti = self.get_data(
            self.ti_var,
            FC.STATE_TURBINE,
            lookup="sf",
            fdata=fdata,
            selection=st_sel,
        )

        k = fdata.get(
            self.k_var,
            np.zeros((n_states, n_turbines), dtype=config.dtype_double),
        )

        k[st_sel] = kti * ti + kb

        return {self.k_var: k}
