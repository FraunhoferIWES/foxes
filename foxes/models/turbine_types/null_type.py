from __future__ import annotations
# mypy: disable-error-code=override

from foxes.core import TurbineType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData


class NullType(TurbineType):
    """
    A turbine type that does not compute any data.
    """

    def __init__(
        self,
        *args: Any,
        needs_rews2: bool = False,
        needs_rews3: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        args
            Additional parameters for TurbineType class
        needs_rews2
            Flag for runs that require the REWS2 variable
        needs_rews3
            Flag for runs that require the REWS3 variable
        kwargs
            Additional parameters for TurbineType class
        """
        super().__init__(*args, **kwargs)
        self._rews2 = needs_rews2
        self._rews3 = needs_rews3

    def needs_rews2(self) -> bool:
        """
        Returns flag for requiring REWS2 variable

        Returns
        -------
        flag
            True if REWS2 is required

        """
        return self._rews2

    def needs_rews3(self) -> bool:
        """
        Returns flag for requiring REWS3 variable

        Returns
        -------
        flag
            True if REWS3 is required

        """
        return self._rews3

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
        return []

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
        st_sel
            The state-turbine selection,
            shape: (n_states, n_turbines)

        Returns
        -------
        results
            The resulting data, keys: output variable str.
            Values

        """
        self.ensure_output_vars(algo, fdata)
        return {}
