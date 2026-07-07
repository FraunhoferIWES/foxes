from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
from typing import TYPE_CHECKING

from foxes.core import TurbineModel
import foxes.variables as FV

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData


class YAWM2YAW(TurbineModel):
    """
    Calculates absolute yaw (i.e. YAWM) from delta
    yaw (i.e. YAWM)

    :group: models.turbine_models

    """

    def output_farm_vars(self, algo: Algorithm) -> list[str]:
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
        return [FV.YAW]

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
        self.ensure_output_vars(algo, fdata)

        yawm = fdata[FV.YAWM][st_sel]
        wd = fdata[FV.WD][st_sel]

        yaw = fdata[FV.YAW]
        yaw[st_sel] = np.mod(wd + yawm, 360.0)

        return {FV.YAW: yaw}
