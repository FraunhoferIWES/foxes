from __future__ import annotations
# mypy: disable-error-code=override

from foxes.core import TurbineModel
import foxes.variables as FV
from foxes.utils import delta_wd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData


class YAW2YAWM(TurbineModel):
    """
    Calculates delta yaw (i.e. YAWM) from absolute
    yaw (i.e. YAW)


    """

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
        return [FV.YAWM]

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

        yaw = fdata[FV.YAW][st_sel]
        wd = fdata[FV.WD][st_sel]

        yawm = fdata[FV.YAWM]
        yawm[st_sel] = delta_wd(wd, yaw)

        return {FV.YAWM: yawm}
