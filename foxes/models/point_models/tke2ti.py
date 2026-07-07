from __future__ import annotations
# mypy: disable-error-code=override

from foxes.core import PointDataModel
from foxes.utils import tke2ti
import foxes.variables as FV
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData


class TKE2TI(PointDataModel):
    """
    Calculates TI from TKE, using TI = sqrt( 3/2 * TKE) / WS

    :group: models.point_models

    """

    def output_point_vars(self, algo: Algorithm) -> list[str]:
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
        return [FV.TI]

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
    ) -> dict[str, object]:
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
        tdata: foxes.core.TData
            The target point data

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_points)

        """
        tke = tdata[FV.TKE]
        ws = tdata[FV.WS]

        ti = tke2ti(tke, ws)

        return {FV.TI: ti}
