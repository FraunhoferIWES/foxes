from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any

from foxes.core import Output, FData
import foxes.constants as FC
import foxes.variables as FV

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.utils import Dict


class FarmOrderedFData(Output):
    """
    Recovers the original farm order for fdata.
    """

    def __init__(
        self,
        variables: list[str] | None = None,
        ambient_vars: bool = True,
        waked_vars: bool = True,
        **kwargs: Any,
    ):
        """
        Parameters
        ----------
        variables
            The farm results variables, or all by default
        ambient_vars
            Include ambient variables
        waked_vars
            Include waked variables
        **kwargs
            Additional parameters for the base class
        """
        super().__init__(**kwargs)
        self.variables = variables
        self.ambient_vars = ambient_vars
        self.waked_vars = waked_vars

    def __repr__(self) -> str:
        vrs = "[all variables]"
        if self.variables is not None:
            vrs = []
            for v in self.variables:
                if not self.ambient_vars and v in FV.amb2var:
                    continue
                if not self.waked_vars and v in FV.var2amb:
                    continue
                vrs.append(v)
            vrs = "[" + ", ".join(vrs) + "]"
        return f"{type(self).__name__}() " + vrs

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        model_results: Any,
        chunk_results: Dict | None = None,
    ) -> Dict:
        """
        The chunk based output calculation.

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
        model_results
            The results from the model calculation for this chunk
        chunk_results
            Existing chunk results

        Returns
        -------
        chunk_results
            The updated chunk results

        """
        chunk_results = super().calculate(
            algo=algo,
            mdata=mdata,
            fdata=fdata,
            tdata=tdata,
            model_results=model_results,
            chunk_results=chunk_results,
        )
        if FC.FDATA_FARM_ORDERED not in chunk_results:
            print("HERE CALC FDATA", mdata.chunki_states)
            order = fdata[FV.ORDER]
            ssel = np.broadcast_to(np.arange(order.shape[0])[:, None], order.shape)
            order_inv = np.empty_like(order)
            order_inv[ssel, order] = np.arange(order.shape[1])[None, :]

            vrs: list[str] = (
                self.variables if self.variables is not None else list(fdata.keys())
            )
            data = {}
            for v in vrs:
                if not self.ambient_vars and v in FV.amb2var:
                    continue
                if not self.waked_vars and v in FV.var2amb:
                    continue
                if (
                    v != FV.ORDER
                    and fdata.dims[v][:2] == (FC.STATE, FC.TURBINE)
                    and fdata[v].shape[1] > 1
                    and np.any(fdata[v] != fdata[v][0, 0, None, None])
                ):
                    data[v] = fdata[v][ssel, order_inv]
                else:
                    data[v] = fdata[v]

            chunk_results[FC.FDATA_FARM_ORDERED] = FData(
                data=data,
                dims={v: fdata.dims[v] for v in data},
            )

        return chunk_results
