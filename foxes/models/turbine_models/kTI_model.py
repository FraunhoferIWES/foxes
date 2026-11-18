from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
from typing import TYPE_CHECKING, Any

from foxes.core import TurbineModel
from foxes.config import config
from foxes.utils.factory import _reject_legacy_wake_coefficients
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData


class kTI(TurbineModel):
    """Wake growth from ``FV.K = FV.K_0 + FV.K_1 * TI``.

    Uniform coefficients override farm data. The intercept defaults to zero
    when neither a uniform value nor ``FV.K_0`` farm data is available.

    Raises
    ------
    ValueError
        If ``FV.K_1`` or the selected turbulence field is unavailable during
        calculation.
    DeprecationWarning
        If removed coefficient keywords are supplied, raised as an exception
        with migration guidance.
    """

    def __init__(
        self,
        k_0: float | None = None,
        k_1: float | None = None,
        ti_var: str = FV.TI,
        ti_val: float | None = None,
        k_var: str = FV.K,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        k_0
            Uniform dimensionless intercept, or ``None`` to use ``FV.K_0``
            from farm data, defaulting to zero if absent.
        k_1
            Uniform dimensionless TI multiplier, or ``None`` to retrieve
            ``FV.K_1`` from farm data.
        ti_var
            Turbulence field, such as ``FV.TI`` or ``FV.AMB_TI``.
        ti_val
            Uniform dimensionless turbulence intensity, or ``None`` to
            retrieve ``ti_var`` from farm data.
        k_var
            Output wake growth field with dimensions ``(FC.STATE, FC.TURBINE)``.
        kwargs
            Removed keywords are rejected; no extra configuration is accepted.
        """
        _reject_legacy_wake_coefficients(kwargs)
        if kwargs:
            raise TypeError(f"kTI: Unexpected parameters {sorted(kwargs)}")
        super().__init__()

        self.ti_var = ti_var
        self.k_var = k_var
        setattr(self, ti_var, ti_val)
        setattr(self, FV.K_0, k_0)
        setattr(self, FV.K_1, k_1)

    def __repr__(self) -> str:
        k_0 = getattr(self, FV.K_0)
        k_1 = getattr(self, FV.K_1)
        ti = getattr(self, self.ti_var)
        tiv = f", ti_val={ti}" if ti is not None else ""
        a = f"k_0={k_0}, k_1={k_1}, ti_var={self.ti_var}{tiv}, k_var={self.k_var}"
        return f"{type(self).__name__}({a})"

    def output_farm_vars(self, algo: Algorithm) -> list[str]:
        """List the wake growth field modified by this model.

        Parameters
        ----------
        algo
            The calculation algorithm.

        Returns
        -------
        output_vars
            The configured ``k_var`` output field.

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
            Farm data containing ``FV.K_0``, ``FV.K_1`` and ``ti_var`` unless
            supplied uniformly. The ``k_var`` growth field is updated in place.
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
        k_1 = self.get_data(
            FV.K_1,
            FC.STATE_TURBINE,
            lookup="sf",
            fdata=fdata,
            selection=st_sel,
        )
        k_0 = self.get_data(
            FV.K_0,
            FC.STATE_TURBINE,
            lookup="sf",
            fdata=fdata,
            selection=st_sel,
            accept_none=True,
        )
        if k_0 is None:
            k_0 = 0.0
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

        k[st_sel] = k_0 + k_1 * ti

        return {self.k_var: k}
