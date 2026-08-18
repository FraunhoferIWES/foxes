from __future__ import annotations
# mypy: disable-error-code=override

from typing import TYPE_CHECKING, Any
import numpy as np

from foxes.core import FarmDataModel

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData


class URelax(FarmDataModel):
    """
    Apply under-relaxation to selected variables
    during iterations.

    Attributes
    ----------
    urel
        The variables and their under-relaxation
        factors between 0 and 1


    """

    def __init__(self, **urel: Any) -> None:
        """
        Constructor.

        Parameters
        ----------
        urel
            The variables and their under-relaxation
            factors between 0 and 1

        """
        super().__init__()
        self.urel = urel
        self.name += "_" + "_".join(list(urel.keys()))

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
        return list(self.urel.keys())

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        res: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
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

        Returns
        -------
        results
            The resulting data, keys: output variable str.
            Values with shape (n_states, n_turbines)

        """
        i0 = fdata.states_i0(counter=True)
        assert i0 is not None
        assert fdata.n_states is not None
        i1 = i0 + fdata.n_states
        pres = algo.prev_farm_results

        cur = fdata if res is None else res
        out = {}
        for v, u in self.urel.items():
            if u > 0 and pres is not None:
                odata = pres[v].to_numpy()[i0:i1]
                out[v] = u * odata + (1 - u) * cur[v]
            else:
                out[v] = cur[v]

        return out
