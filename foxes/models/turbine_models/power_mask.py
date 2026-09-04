from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
from typing import TYPE_CHECKING, cast

from foxes.core import Model, TurbineModel
from foxes.config import config
from foxes.utils import cubic_roots
import foxes.variables as FV

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData
    from foxes.core.axial_induction_model import AxialInductionModel
    from foxes.core.model import LoadedData


class PowerMask(TurbineModel):
    """
    Invokes a maximal power value.

    This may correspond to turbine derating, if
    the maximal power value is below rated power.
    For higher values, a boost is introduced.

    The model updates the P and CT variables,
    so it is wise to use it after calling the
    turbine type model.
    """

    def __init__(
        self,
        var_ws_P: str = FV.REWS3,
        factor_P: float = 1.0e3,
        P_lim: float = 100,
        induction: str | AxialInductionModel = "Betz",
    ) -> None:
        """
        Parameters
        ----------
        var_ws_P
            The wind speed variable for power lookup
        factor_P
            The power unit factor, e.g. 1000 for kW
        P_lim
            Threshold power delta for boosts
        induction
            The induction model
        """
        super().__init__()

        self.var_ws_P = var_ws_P
        self.factor_P = factor_P
        self.P_lim = P_lim
        self.induction = induction
        self._P_rated: np.ndarray | None = None

    def __repr__(self) -> str:
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        a = f"var_ws_P={self.var_ws_P}, P_lim={self.P_lim}, induction={iname}"
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
        return [FV.P, FV.CT]

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls
            All sub models

        """
        return [cast(Model, self.induction)]

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
        if isinstance(self.induction, str):
            self.induction = algo.mbook.axial_induction[self.induction]
        loaded_data = super().initialize(
            algo, loaded_data=loaded_data, force=force, verbosity=verbosity
        )

        p_rated: list[float] = []
        turbine_types = algo.farm_controller.turbine_types
        assert turbine_types is not None
        for t in turbine_types:
            Pnom = config.dtype_double(t.P_nominal)
            if np.isnan(Pnom):
                raise ValueError(
                    f"Model '{self.name}': P_nominal is NaN for turbine type '{t.name}'"
                )
            p_rated.append(float(Pnom))
        self._P_rated = np.array(p_rated, dtype=config.dtype_double)
        return loaded_data

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
        # prepare:
        self.ensure_output_vars(algo, fdata)
        P = fdata[FV.P]
        max_P = fdata[FV.MAX_P]
        P_rated0 = self._P_rated
        assert P_rated0 is not None, "Rated powers not initialized"
        P_rated = P_rated0[None, :]

        # select power entries for which this is active:
        n_states = fdata.n_states
        n_turbines = fdata.n_turbines
        assert n_states is not None and n_turbines is not None
        sel = np.zeros((n_states, n_turbines), dtype=np.bool_)
        sel[st_sel] = True
        sel = (
            sel
            & ~np.isnan(max_P)
            & (
                ((max_P < P_rated) & (P > max_P))
                | ((max_P > P_rated) & (P > P_rated - self.P_lim))
            )
        )
        if np.any(sel):
            # apply selection:
            assert not isinstance(self.induction, str), (
                "Induction model not initialized"
            )
            max_P = max_P[sel]
            ws = fdata[self.var_ws_P][sel]
            rho = fdata[FV.RHO][sel]
            r = fdata[FV.D][sel] / 2
            P = P[sel]
            ct = fdata[FV.CT][sel]

            # calculate power efficiency e of turbine
            # e is the ratio of the cp derived from the power curve
            # and the theoretical cp from the turbine induction
            cp = P / (0.5 * ws**3 * rho * np.pi * r**2) * self.factor_P
            a = self.induction.ct2a(ct)
            cp_a = 4 * a**3 - 8 * a**2 + 4 * a
            e = cp / cp_a
            del cp, a, cp_a, ct, P

            # calculating new cp for changed power
            cp = max_P / (0.5 * ws**3 * rho * np.pi * r**2) * self.factor_P

            # find roots:
            N = len(cp)
            a3: np.ndarray = np.full(N, 4.0, dtype=config.dtype_double)
            a2: np.ndarray = np.full(N, -8.0, dtype=config.dtype_double)
            a1: np.ndarray = np.full(N, 4.0, dtype=config.dtype_double)
            a0 = -cp / e
            rts = cubic_roots(a0, a1, a2, a3)
            rts[np.isnan(rts)] = np.inf
            rts[rts <= 0.0] = np.inf
            a = np.min(rts, axis=1)
            del a0, a1, a2, a3, rts

            # set results:
            P = fdata[FV.P]
            ct = fdata[FV.CT]
            P[sel] = max_P
            ct[sel] = 4 * a * (1 - a)

        return {FV.P: fdata[FV.P], FV.CT: fdata[FV.CT]}
