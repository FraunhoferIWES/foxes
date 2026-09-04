from __future__ import annotations
# mypy: disable-error-code=override

from foxes.core import TurbineModel
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData


class Calculator(TurbineModel):
    """
    Calculates variables based on given functions.

        Beware that the turbine ordering in fdata is in downwind order,
        hence external data X of shape (n_states, n_turbines) in farm order
        needs to be reordered by X[ssel, order] with
        ssel = fdata[FV.ORDER_SSEL], order = fdata[FV.ORDER]
        before using it in combination with fdata variables.
    """

    def __init__(
        self,
        in_vars: list[str],
        out_vars: list[str],
        func: Any,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        in_vars
            The input farm variables
        out_vars
            The output variables
        func: Function
            The function: f(in0, in1, ..., algo, mdata, fdata, st_sel) -> (out0, out1, ...)
            where inX and outY are arrays and
            st_sel is the state-turbine selection slice or array.
            All arrays have shape (n_states, n_turbines).

            Beware that the turbine ordering in fdata is in downwind order,
            hence external data X of shape (n_states, n_turbines) in farm order
            needs to be reordered by X[ssel, order] with
            ssel = fdata[FV.ORDER_SSEL], order = fdata[FV.ORDER]
            before using it in combination with fdata variables.
        kwargs
            Additional arguments for TurbineModel
        """
        super().__init__(**kwargs)
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.func = func

    def __repr__(self) -> str:
        a = f"{self.in_vars}, {self.out_vars}"
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
        return self.out_vars

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
        ins = [fdata[v] if v in fdata else mdata[v] for v in self.in_vars]
        outs = self.func(*ins, algo=algo, mdata=mdata, fdata=fdata, st_sel=st_sel)

        return {v: outs[vi] for vi, v in enumerate(self.out_vars)}
