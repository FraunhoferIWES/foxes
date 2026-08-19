from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
from typing import TYPE_CHECKING

from foxes.core import TurbineModel, TData
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData
    from foxes.core.model import LoadedData
    from foxes.core.model import Model


class RotorCentreCalc(TurbineModel):
    """
    Calculates data at the rotor centre

    Attributes
    ----------
    calc_vars
        The variables that are calculated by the model,
        keys: var names, values: rotor var names


    """

    def __init__(self, calc_vars: dict[str, str] | list[str]) -> None:
        """
        Constructor.

        Parameters
        ----------
        calc_vars
            The variables that are calculated by the model,
            keys: var names, values: rotor var names

        """
        super().__init__()

        if isinstance(calc_vars, dict):
            self.calc_vars = calc_vars
        else:
            self.calc_vars = {v: v for v in calc_vars}

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """
        Initializes the model.

        Parameters
        ----------
        algo
            The calculation algorithm
        loaded_data
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        loaded_data
            The loaded data, containing keys "coords", "data_vars", and "extra_data".
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.

        """
        self._wcalc = algo.get_model("PointWakesCalculation")()
        return super().initialize(
            algo, loaded_data=loaded_data, force=force, verbosity=verbosity
        )

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls
            Names of all sub models

        """
        return [self._wcalc]

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
        return list(self.calc_vars.keys())

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

        # prepare target point data:
        tdata = TData.from_points(
            fdata[FV.TXYH],
            data={
                v: np.zeros_like(fdata[FV.X][:, :, None])
                for v in self.calc_vars.values()
            },
            dims={v: (FC.STATE, FC.TARGET, FC.TPOINT) for v in self.calc_vars.values()},
            name=f"{self.name}_tdata",
        )

        # run ambient calculation:
        res = algo.states.calculate(algo, mdata, fdata, tdata)
        for v, a in FV.var2amb.items():
            if v in res:
                res[a] = res[v].copy()
        tdata.update(res)

        # run wake calculation:
        res = self._wcalc.calculate(algo, mdata, fdata, tdata)

        # extract results:
        out = {v: fdata[v] for v in self.calc_vars.keys()}
        for v in out.keys():
            w = self.calc_vars[v]
            out[v][st_sel] = res[w][st_sel][..., 0]

        return out
