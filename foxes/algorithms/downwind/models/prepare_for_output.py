from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
from typing import TYPE_CHECKING

from foxes.core import DataCalcModel
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.output import Output


class PrepareForOutput(DataCalcModel):
    """
    Filters mdata, fdata, tdata for output requirements
    and re-orders variables from downwind to farm order.
    """

    def __init__(self, output: Output | None) -> None:
        """
        Parameters
        ----------
        output
            The output object
        """
        super().__init__()
        self.output = output

    def output_coords(self) -> tuple[str, ...]:
        """
        Gets the coordinates of all output arrays

        Returns
        -------
        dims
            The coordinates of all output arrays

        """
        raise NotImplementedError
    
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
        tdata: TData | None = None,
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
            Empty dummy results, data is changed on the fly

        """
        # case not output:
        if self.output is None:
            for data in (mdata, fdata, tdata):
                if data is not None:
                    for v in list(data.keys()):
                        del data[v], data.dims[v]
            return {}

        # get variables:
        vrs_mdata = self.output.mdata_variables
        vrs_fdata = self.output.fdata_variables
        vrs_tdata = self.output.tdata_variables
        
        # loop over targets:
        order = fdata[FV.ORDER]
        ssel = None
        order_inv = None
        for (vrs, data) in zip((vrs_mdata, vrs_fdata, vrs_tdata), (mdata, fdata, tdata)):
            if data is None:
                continue
            all_vrs = list(set(data.keys()) - set(data.sizes.keys()))
            if vrs is None:
                vrs = all_vrs
            for v in all_vrs:
                dta = data.pop(v)
                dms = data.dims.pop(v)
                if v in vrs:
                    if FC.STATE in dms and FC.TURBINE in dms:
                        assert dms[:2] == (FC.STATE, FC.TURBINE), (
                            f"{self.name}: Require dimensions of '{v}' to match ({FC.STATE}, {FC.TURBINE}, ...), got {dms}"
                        )

                        # reoreder to farm order, if needed:
                        if (
                            v != FV.ORDER
                            and dta.shape[1] > 1
                            and np.any(dta != dta[0, 0, None, None])
                        ):
                            if ssel is None:
                                ssel = np.broadcast_to(np.arange(order.shape[0])[:, None], order.shape)
                                order_inv = np.empty_like(order)
                                order_inv[ssel, order] = np.arange(order.shape[1])[None, :]
                            dta = dta[ssel, order_inv, ...]

                    data[v] = dta
                    data.dims[v] = dms

        return {}
