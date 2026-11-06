from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from foxes.core import TurbineModel
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core import Algorithm
    from foxes.core.data import FData, MData

class TurbineEvaluation(TurbineModel):
    """
    Final performance evaluation on the turbine level,
    in terms of capacity, efficiency, yield etc.
    """

    def __init__(
        self, 
        variables: list[str] = [], 
        name="TurbineEvaluation", 
        **kwargs,
    ):
        """
        Parameters
        ----------
        variables
            The list of variables to evaluate, will be merged
            with the variables given to the calculate function.
        name
            The name of the turbine model
        **kwargs
            Additional keyword arguments passed to the base class.
        """
        self.variables = variables
        super().__init__(name=name, **kwargs)

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
        return self.variables

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        st_sel: slice | np.ndarray = slice(None),
        variables: list[str] = [],
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
            for shape: (n_states, n_turbines)
        variables
            The list of variables to calculate, will be merged
            with the variables defined in the model.

        Returns
        -------
        results
            The resulting data, keys: output variable str.
            Values

        """
        # prepare:
        self.ensure_output_vars(algo, fdata)
        variables = list(set(self.variables) | set(variables))
        results: dict[str, np.ndarray] = {}

        n_states = fdata.n_states
        n_turbines = fdata.n_turbines
        assert n_states is not None and n_turbines is not None
        sel_data: np.ndarray | None
        if isinstance(st_sel, slice) and st_sel == slice(None):
            sel_data = None
        elif isinstance(st_sel, slice):
            sel_arr = np.zeros((n_states, n_turbines), dtype=np.bool_)
            sel_arr[st_sel] = True
            sel_data = sel_arr
        else:
            sel_data = st_sel

        # commpute variables:
        for v in variables:

            # calculate efficiency:
            if v == FV.EFF:
                P = self.get_data(
                    FV.P,
                    FC.STATE_TURBINE,
                    lookup="sf",
                    fdata=fdata,
                    upcast=False,
                    selection=sel_data,
                )
                results[FV.EFF]

            results[var] = fdata.get(var, np.zeros((fdata.n_states, fdata.n_turbines), dtype=np.float64))

        

        return results