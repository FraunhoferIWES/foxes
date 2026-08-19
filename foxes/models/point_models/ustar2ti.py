from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
from foxes.core import PointDataModel
from foxes.utils import ustar2ti
import foxes.variables as FV
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData


class Ustar2TI(PointDataModel):
    """
    Calculates TI from Ustar, using TI = Ustar / (kappa*WS)

    Attributes
    ----------
    max_ti
        Upper limit of the computed TI values


    """

    def __init__(self, max_ti: float | None = None, **kwargs: Any) -> None:
        """
        Constructor

        Parameters
        ----------
        max_ti
            Upper limit of the computed TI values
        kwargs
            Additional parameters for the base class

        """
        super().__init__(**kwargs)
        self.max_ti = max_ti

    def output_point_vars(self, algo: Algorithm) -> list[str]:
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
        return [FV.TI]

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
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
        tdata
            The target point data

        Returns
        -------
        results
            The resulting data, keys: output variable str.
            Values

        """
        ustar = tdata[FV.USTAR]
        ws = tdata[FV.WS]

        ti = ustar2ti(ustar, ws, self.max_ti)

        return {FV.TI: ti}
